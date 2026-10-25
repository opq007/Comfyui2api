from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from .backend_store import BackendStore, InstanceRecord
from .comfy_client import ComfyUIClient
from .config import Config
from .routing import ordered_pool, pick_instance
from .util import utc_now_unix


logger = logging.getLogger(__name__)

HEALTH_UNKNOWN = "unknown"
HEALTH_HEALTHY = "healthy"
HEALTH_UNHEALTHY = "unhealthy"
HEALTH_DISABLED = "disabled"

# While an instance is not yet HEALTHY (starting up / recovering), probe far
# more aggressively than the configured steady-state interval so recovery is
# detected in seconds instead of a full `health_check_interval_s`.
PROBE_FAST_INTERVAL_S = 5.0
# Minimum spacing between on-demand probe kicks for the same instance, so the
# job scheduler cannot turn kicks into a request-per-second health hammer.
KICK_THROTTLE_S = 5.0
# Poll spacing used by the deferred job-client close loop.
DELAYED_CLOSE_POLL_S = 1.0


@dataclass
class InstanceRuntime:
    slug: str
    health: str = HEALTH_UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_at: int | None = None
    last_error: str | None = None
    in_flight: int = 0
    health_client: ComfyUIClient | None = None
    job_client: ComfyUIClient | None = None
    probe_task: asyncio.Task[None] | None = None
    generation: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "health": self.health,
            "consecutive_failures": self.consecutive_failures,
            "last_check_at": self.last_check_at,
            "last_error": self.last_error,
            "in_flight": self.in_flight,
        }


class InstancePool:
    def __init__(self, *, cfg: Config, store: BackendStore) -> None:
        self.cfg = cfg
        self.store = store
        self._lock = asyncio.Lock()
        self._runtimes: dict[str, InstanceRuntime] = {}
        self._cursors: dict[str, str | None] = {}
        self._probe_kick_at: dict[str, float] = {}
        self._delayed_close: set[asyncio.Task[None]] = set()
        # Slot events start CLEARED: an event only signals a real transition (a
        # slot was freed / an instance recovered). Pre-setting it made
        # `wait_for_slot` return immediately forever, turning the job scheduler
        # into a busy-spin idle loop.
        self._slot_event = asyncio.Event()
        self._rng = random.Random()

    def _reset_slot_event(self) -> None:
        self._slot_event = asyncio.Event()

    async def start(self) -> None:
        self._reset_slot_event()
        instances = await self.store.list_instances()
        for record in instances:
            await self.sync_instance(record, probe_now=record.enabled)

    async def aclose(self) -> None:
        async with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
        for runtime in runtimes:
            runtime.generation += 1
            if runtime.probe_task is not None:
                runtime.probe_task.cancel()
            if runtime.health_client is not None:
                await runtime.health_client.aclose()
            if runtime.job_client is not None:
                await runtime.job_client.aclose()
        pending_close = list(self._delayed_close)
        for task in pending_close:
            task.cancel()
        if pending_close:
            await asyncio.gather(*pending_close, return_exceptions=True)

    async def sync_instance(self, record: InstanceRecord, *, probe_now: bool) -> None:
        recreate_client = False
        async with self._lock:
            runtime = self._runtimes.get(record.slug)
            if runtime is None:
                runtime = InstanceRuntime(slug=record.slug)
                self._runtimes[record.slug] = runtime
                recreate_client = True
            else:
                current = runtime.health_client
                if (
                    current is None
                    or current.base_url != record.base_url
                    or (current.auth_token or None) != (record.auth_token or None)
                ):
                    recreate_client = True
            if recreate_client:
                old_health = runtime.health_client
                old_job = runtime.job_client
                runtime.generation += 1
                if runtime.probe_task is not None:
                    runtime.probe_task.cancel()
                    runtime.probe_task = None
                runtime.health_client = ComfyUIClient(
                    record.base_url,
                    http_timeout_s=self.cfg.health_check_timeout_s,
                    auth_token=record.auth_token,
                )
                runtime.job_client = ComfyUIClient(
                    record.base_url,
                    http_timeout_s=self.cfg.http_timeout_s,
                    auth_token=record.auth_token,
                )
                if old_health is not None:
                    await old_health.aclose()
                if old_job is not None:
                    # In-flight jobs still hold references to the old client;
                    # closing it now would kill them with "client is closed".
                    # Defer the close until the instance goes idle (or a
                    # deadline expires as a leak guard).
                    task = asyncio.create_task(self._close_client_when_idle(old_job, record.slug))
                    self._delayed_close.add(task)
                    task.add_done_callback(self._delayed_close.discard)
            if not record.enabled:
                runtime.generation += 1
                if runtime.probe_task is not None:
                    runtime.probe_task.cancel()
                    runtime.probe_task = None
                runtime.health = HEALTH_DISABLED
                runtime.consecutive_failures = 0
                runtime.consecutive_successes = 0
                runtime.last_error = None
            elif runtime.health == HEALTH_DISABLED or recreate_client:
                runtime.health = HEALTH_UNKNOWN
                runtime.consecutive_failures = 0
                runtime.consecutive_successes = 0
                runtime.last_error = None
            should_loop = record.enabled
            generation = runtime.generation
            client = runtime.health_client
        if should_loop and client is not None:
            await self._ensure_probe_loop(record, generation=generation, probe_now=probe_now)

    async def drop_instance(self, slug: str) -> None:
        async with self._lock:
            runtime = self._runtimes.pop(slug, None)
        if runtime is None:
            return
        runtime.generation += 1
        if runtime.probe_task is not None:
            runtime.probe_task.cancel()
        if runtime.health_client is not None:
            await runtime.health_client.aclose()
        if runtime.job_client is not None:
            await runtime.job_client.aclose()

    async def runtime_snapshot(self, slug: str) -> dict[str, Any]:
        async with self._lock:
            runtime = self._runtimes.get(slug)
            if runtime is None:
                return {
                    "health": HEALTH_UNKNOWN,
                    "consecutive_failures": 0,
                    "last_check_at": None,
                    "last_error": None,
                    "in_flight": 0,
                }
            return runtime.snapshot()

    async def healthy_slugs(self, slugs: list[str]) -> list[str]:
        async with self._lock:
            return [slug for slug in slugs if self._is_healthy_locked(slug)]

    async def ready_slugs(self, slugs: list[str], *, exclude: set[str] | None = None) -> list[str]:
        records = {item.slug: item for item in await self.store.list_instances()}
        async with self._lock:
            return self._ready_locked(slugs, records=records, skipped=exclude or set())

    def _ready_locked(
        self,
        slugs: list[str],
        *,
        records: dict[str, InstanceRecord],
        skipped: set[str],
    ) -> list[str]:
        ready: list[str] = []
        for slug in ordered_pool(slugs):
            if slug in skipped:
                continue
            record = records.get(slug)
            runtime = self._runtimes.get(slug)
            if record is None or runtime is None:
                continue
            if not record.enabled or runtime.health != HEALTH_HEALTHY:
                continue
            if runtime.in_flight >= record.max_in_flight:
                continue
            ready.append(slug)
        return ready

    async def choose_and_reserve(
        self,
        *,
        model_slug: str,
        policy: str,
        pool_slugs: list[str],
        exclude: set[str] | None = None,
    ) -> str | None:
        # The readiness check and the in_flight increment run under the same
        # lock acquisition: checking first and reserving in a second critical
        # section allowed concurrent dispatches to oversubscribe
        # max_in_flight.
        records = {item.slug: item for item in await self.store.list_instances()}
        async with self._lock:
            ready = self._ready_locked(pool_slugs, records=records, skipped=exclude or set())
            if not ready:
                return None
            cursor = self._cursors.get(model_slug)
            chosen, next_cursor = pick_instance(policy=policy, ready_slugs=ready, cursor=cursor, rng=self._rng)
            if chosen is None:
                return None
            runtime = self._runtimes.get(chosen)
            if runtime is None:
                return None
            runtime.in_flight += 1
            if policy == "round_robin":
                self._cursors[model_slug] = next_cursor
            self._slot_event.clear()
            return chosen

    async def release(self, slug: str | None) -> None:
        if not slug:
            return
        async with self._lock:
            runtime = self._runtimes.get(slug)
            if runtime is not None and runtime.in_flight > 0:
                runtime.in_flight -= 1
            self._slot_event.set()

    async def client_for(self, slug: str) -> ComfyUIClient:
        async with self._lock:
            runtime = self._runtimes.get(slug)
            if runtime is None or runtime.job_client is None:
                raise LookupError(f"Instance `{slug}` has no live client.")
            return runtime.job_client

    async def wait_for_slot(self, timeout_s: float) -> bool:
        try:
            await asyncio.wait_for(self._slot_event.wait(), timeout=max(0.05, timeout_s))
            return True
        except asyncio.TimeoutError:
            return False

    def _is_healthy_locked(self, slug: str) -> bool:
        runtime = self._runtimes.get(slug)
        return runtime is not None and runtime.health == HEALTH_HEALTHY

    async def _ensure_probe_loop(self, record: InstanceRecord, *, generation: int, probe_now: bool) -> None:
        async with self._lock:
            runtime = self._runtimes.get(record.slug)
            if runtime is None:
                return
            if runtime.probe_task is None or runtime.probe_task.done():
                runtime.probe_task = asyncio.create_task(
                    self._probe_loop(record.slug, generation, probe_first=not probe_now),
                    name=f"health-{record.slug}",
                )
        if probe_now:
            await self._probe_once(record.slug, generation)

    async def kick_probes(self, slugs: list[str]) -> None:
        """Request an immediate health probe for the given instances.

        Called by the job scheduler when queued jobs are blocked because no
        instance is ready, so recovery is detected right away instead of
        waiting out the fixed probe interval. Throttled per slug: a kick
        within `KICK_THROTTLE_S` of the previous one is a no-op, so a busy
        scheduler cannot turn this into a request-per-second hammer.
        """
        now = time.monotonic()
        targets: list[tuple[str, int]] = []
        async with self._lock:
            for slug in slugs:
                runtime = self._runtimes.get(slug)
                if runtime is None:
                    continue
                if now - self._probe_kick_at.get(slug, 0.0) < KICK_THROTTLE_S:
                    continue
                self._probe_kick_at[slug] = now
                targets.append((slug, runtime.generation))
        for slug, generation in targets:
            task = asyncio.create_task(self._probe_once(slug, generation), name=f"health-kick-{slug}")
            # Fire-and-forget: probe failures are already logged in _probe_once.
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _probe_loop(self, slug: str, generation: int, *, probe_first: bool = True) -> None:
        # Probe FIRST, then sleep: sleeping first meant that after the startup
        # probe missed (backend still booting), the next attempt waited a full
        # health_check_interval_s even if the backend became ready seconds
        # later — the "first task polls emptily" window.
        first = True
        while True:
            try:
                record = await self.store.get_instance(slug)
                if record is None or not record.enabled:
                    return
                async with self._lock:
                    runtime = self._runtimes.get(slug)
                    if runtime is None or runtime.generation != generation:
                        return
                    health = runtime.health
                # Skip only the very first probe when the caller already ran
                # one synchronously via probe_now.
                if not first or probe_first:
                    await self._probe_once(slug, generation)
                first = False
                if health == HEALTH_HEALTHY:
                    interval = record.health_interval_s or self.cfg.health_check_interval_s
                    delay_s: float = float(max(1, int(interval)))
                else:
                    delay_s = PROBE_FAST_INTERVAL_S
                await asyncio.sleep(delay_s)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transient store/DB error must not kill the probe loop and
                # freeze this instance's health state forever.
                logger.exception("health probe loop error: instance=%s", slug)
                await asyncio.sleep(PROBE_FAST_INTERVAL_S)

    async def _probe_once(self, slug: str, generation: int) -> None:
        record = await self.store.get_instance(slug)
        if record is None or not record.enabled:
            return
        async with self._lock:
            runtime = self._runtimes.get(slug)
            if runtime is None or runtime.generation != generation or runtime.health_client is None:
                return
            client = runtime.health_client
        error: str | None = None
        ok = False
        try:
            await asyncio.wait_for(client.system_stats(), timeout=self.cfg.health_check_timeout_s)
            ok = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.info("health check failed: instance=%s error=%s", slug, error)
        needs_warm = False
        job_client: ComfyUIClient | None = None
        async with self._lock:
            runtime = self._runtimes.get(slug)
            if runtime is None or runtime.generation != generation:
                return
            runtime.last_check_at = utc_now_unix()
            if ok:
                runtime.consecutive_successes += 1
                runtime.consecutive_failures = 0
                runtime.last_error = None
                was_healthy = runtime.health == HEALTH_HEALTHY
                if runtime.consecutive_successes >= self.cfg.health_check_recovery_threshold:
                    runtime.health = HEALTH_HEALTHY
                    self._slot_event.set()
                    job_client = runtime.job_client
                    needs_warm = (
                        not was_healthy
                        and job_client is not None
                        and not job_client.object_info_cached
                    )
            else:
                runtime.consecutive_failures += 1
                runtime.consecutive_successes = 0
                runtime.last_error = error
                if runtime.health == HEALTH_HEALTHY:
                    runtime.health = HEALTH_UNKNOWN
                if runtime.consecutive_failures >= self.cfg.health_check_fail_threshold:
                    runtime.health = HEALTH_UNHEALTHY
                elif runtime.health != HEALTH_UNHEALTHY:
                    runtime.health = HEALTH_UNKNOWN
        if needs_warm:
            # The first job on a fresh gateway used to pay the full /object_info
            # fetch (seconds on large backends) while sitting in "queued" with
            # no progress events. Warm that cache as soon as the instance is
            # known healthy, so the first task dispatches straight through.
            task = asyncio.create_task(self._warm_object_info(slug, generation, job_client), name=f"warm-object-info-{slug}")
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _warm_object_info(self, slug: str, generation: int, client: ComfyUIClient) -> None:
        try:
            await client.object_info()
            logger.info("warmed /object_info cache: instance=%s", slug)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Best-effort prefetch: the first job will retry on demand.
            logger.info("object_info warmup failed (will retry on first job): instance=%s error=%s", slug, exc)

    async def _close_client_when_idle(self, client: ComfyUIClient, slug: str) -> None:
        # Upper bound so a permanently-stuck in_flight count cannot keep a
        # replaced client alive forever.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(60.0, float(getattr(self.cfg, "timeout_s", 3600)))
        try:
            while loop.time() < deadline:
                async with self._lock:
                    runtime = self._runtimes.get(slug)
                    idle = runtime is None or runtime.in_flight <= 0
                if idle:
                    break
                await asyncio.sleep(DELAYED_CLOSE_POLL_S)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await client.aclose()
            except Exception:
                logger.exception("deferred job client close failed: instance=%s", slug)
