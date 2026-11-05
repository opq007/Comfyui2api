from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from websockets.exceptions import WebSocketException

from .comfy_client import ComfyApiError, ComfyUIClient
from .instance_pool import InstancePool
from .util import utc_now_unix


if TYPE_CHECKING:
    from .jobs import Job, JobManager


logger = logging.getLogger(__name__)

STAGED_PREFIX = "staged:"
MAX_REROUTE = 1

# Transport-level failures that should reroute a job to another instance
# instead of failing it. httpx.ConnectError/ReadTimeout and websockets errors
# are neither OSError nor builtin TimeoutError, so without them the reroute
# path never fired for the most common failure (backend just went down).
REROUTABLE_EXCEPTIONS = (
    ComfyApiError,
    OSError,
    TimeoutError,
    FileNotFoundError,
    ValueError,
    httpx.HTTPError,
    WebSocketException,
)


class JobScheduler:
    """Work-conserving dequeue: pick a live instance, reserve a slot, then run."""

    def __init__(self, manager: JobManager) -> None:
        self.manager = manager
        self._queued: deque[str] = deque()
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._running: set[asyncio.Task[None]] = set()

    def enqueue(self, job_id: str) -> None:
        self._queued.append(job_id)
        if self._wake is not None:
            self._wake.set()

    async def start(self) -> None:
        self._wake = asyncio.Event()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="job-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        running = list(self._running)
        for item in running:
            item.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        self._running.clear()

    async def _loop(self) -> None:
        pool: InstancePool = self.manager.pool
        while True:
            try:
                dispatched = await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transient error (e.g. sqlite "database is locked") must not
                # kill this task: nothing restarts it, and every queued job
                # would stay pending forever — silently, without even a
                # timeout failure, because timeout is only checked inside
                # _scan_once.
                logger.exception("job scheduler scan failed; retrying after wake interval")
                dispatched = False
                await asyncio.sleep(max(0.05, float(self.manager.cfg.scheduler_wake_interval_s)))
            if dispatched:
                continue
            if self._wake is None:
                self._wake = asyncio.Event()
            self._wake.clear()
            if not self._queued:
                # Idle: no queued jobs. Block on _wake until a job is actually
                # enqueued — no periodic wake, no busy-spin. This is the idle
                # fast path that previously spun because `wait_for_slot` returned
                # immediately on a pre-set event.
                await self._wake.wait()
                continue
            # Queued jobs but all current instances are busy / unavailable: wait
            # for a slot to free OR a wake, with a bounded re-check interval.
            timeout_s = min(1.0, max(0.05, float(self.manager.cfg.scheduler_wake_interval_s)))
            wait_wake = asyncio.create_task(self._wake.wait())
            wait_slot = asyncio.create_task(pool.wait_for_slot(timeout_s))
            _done, pending = await asyncio.wait({wait_wake, wait_slot}, return_when=asyncio.FIRST_COMPLETED)
            for item in pending:
                item.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _scan_once(self) -> bool:
        blocked_models: set[str] = set()
        blocked_instance_slugs: set[str] = set()
        dispatched = False
        for job_id in list(self._queued):
            job = await self.manager.get_job(job_id)
            if job is None:
                self._remove(job_id)
                continue
            if _timed_out(job, self.manager.cfg.timeout_s):
                self._remove(job_id)
                await self.manager.fail_job(job_id, "Timed out waiting for an available backend.")
                continue
            decision = await self._live_check(job)
            if decision == "fail_missing":
                self._remove(job_id)
                await self.manager.fail_job(job_id, "The requested model is no longer available.")
                continue
            if decision == "fail_disabled":
                self._remove(job_id)
                await self.manager.fail_job(job_id, "The requested model is disabled.")
                continue
            if decision == "fail_workflow":
                self._remove(job_id)
                await self.manager.fail_job(job_id, f"Workflow bound to model `{job.model_slug}` is unavailable.")
                continue
            if job.model_slug in blocked_models:
                continue
            record = await self.manager.backend.get_model(job.model_slug)
            if record is None:
                self._remove(job_id)
                await self.manager.fail_job(job_id, "The requested model is no longer available.")
                continue
            ready = await self.manager.pool.ready_slugs(list(record.instance_slugs))
            if not ready:
                blocked_models.add(job.model_slug)
                blocked_instance_slugs.update(record.instance_slugs)
                continue
            self._remove(job_id)
            task = asyncio.create_task(self._dispatch(job_id), name=f"job-run-{job_id[:8]}")
            self._running.add(task)
            task.add_done_callback(self._running.discard)
            dispatched = True
        if blocked_instance_slugs:
            # Queued jobs are blocked because no instance is ready. Kick the
            # health probes now (throttled inside the pool) instead of letting
            # the scheduler poll emptily until the next fixed probe tick.
            kick = getattr(self.manager.pool, "kick_probes", None)
            if callable(kick):
                await kick(sorted(blocked_instance_slugs))
        return dispatched

    async def _live_check(self, job: Job) -> str:
        record = await self.manager.backend.get_model(job.model_slug)
        if record is None:
            return "fail_missing"
        if not record.enabled:
            return "fail_disabled"
        workflow_name = (record.workflow_name or "").strip()
        if not workflow_name:
            return "fail_workflow"
        workflow = await self.manager.registry.get(workflow_name)
        if workflow is None:
            for item in await self.manager.registry.list():
                if item.name.lower() == workflow_name.lower():
                    workflow = item
                    break
        if workflow is None:
            return "fail_workflow"
        if job.workflow != workflow.name:
            await self.manager._update(job.job_id, workflow=workflow.name)
        return "ok"

    async def _dispatch(self, job_id: str) -> None:
        job = await self.manager.get_job(job_id)
        if job is None:
            return
        exclude: set[str] = set()
        last_error = "No healthy ComfyUI instance available."
        try:
            for attempt in range(MAX_REROUTE + 1):
                record = await self.manager.backend.get_model(job.model_slug)
                if record is None or not record.enabled:
                    await self.manager.fail_job(job_id, "The requested model is no longer available.")
                    return
                slug: str | None = None
                # The reservation itself lives inside the try: from the moment a
                # slot is taken, the `finally` releases it on every exit path.
                # Taking the reservation outside used to leak `in_flight`, and a
                # leaked reservation makes the instance unschedulable for good —
                # `ready_slugs` skips it from then on while the dashboard still
                # reports it healthy, so later jobs queue forever behind a slot
                # nobody will free.
                try:
                    slug = await self.manager.pool.choose_and_reserve(
                        model_slug=job.model_slug,
                        policy=record.routing_policy,
                        pool_slugs=list(record.instance_slugs),
                        exclude=exclude,
                    )
                    if slug is None:
                        if attempt == 0:
                            self.enqueue(job_id)
                        else:
                            await self.manager.fail_job(job_id, last_error)
                        return
                    try:
                        client = await self.manager.pool.client_for(slug)
                    except LookupError as exc:
                        # The instance stopped having a live client between the
                        # reservation and the dispatch (deleted / re-created).
                        # That is a submit-time failure, so spec §11.5 / item 10
                        # apply: release, retry once elsewhere, then fail.
                        last_error = f"{type(exc).__name__}: {exc}"
                        logger.info(
                            "instance has no live client, may reroute: job_id=%s instance=%s attempt=%s error=%s",
                            job_id,
                            slug,
                            attempt,
                            last_error,
                        )
                        exclude.add(slug)
                        if attempt >= MAX_REROUTE:
                            await self.manager.fail_job(job_id, last_error)
                            return
                        continue
                    staged_image = job.image
                    staged_params = dict(job.standard_params)
                    try:
                        uploaded = await upload_staged_values(
                            job=job,
                            client=client,
                            input_subdir=self.manager.cfg.input_subdir,
                        )
                        await self.manager._update(
                            job_id,
                            instance_slug=slug,
                            image=uploaded.get(job.image, job.image),
                            standard_params=_rewrite_params(job.standard_params, uploaded),
                        )
                        await self.manager._run_job(job_id, client=client)
                        return
                    except REROUTABLE_EXCEPTIONS as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        current = await self.manager.get_job(job_id)
                        if current is not None and current.prompt_id:
                            # The prompt was already queued on ComfyUI and this job
                            # is about to be failed: remove the queued entry so the
                            # backend does not keep burning GPU on an abandoned
                            # prompt. Entries already executing are left alone.
                            await _cancel_comfy_prompt(client, current.prompt_id)
                            await self.manager.fail_job(job_id, last_error)
                            return
                        logger.info(
                            "queue_prompt/upload failed, may reroute: job_id=%s instance=%s attempt=%s error=%s",
                            job_id,
                            slug,
                            attempt,
                            last_error,
                        )
                        await self.manager._update(
                            job_id,
                            instance_slug="",
                            image=staged_image,
                            standard_params=staged_params,
                        )
                        job.image = staged_image
                        job.standard_params = staged_params
                        exclude.add(slug)
                        if attempt >= MAX_REROUTE:
                            await self.manager.fail_job(job_id, last_error)
                            return
                finally:
                    if slug is not None:
                        await self.manager.pool.release(slug)
        except Exception as exc:
            logger.exception("job dispatch failed: job_id=%s", job_id)
            await self.manager.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    def _remove(self, job_id: str) -> None:
        try:
            self._queued.remove(job_id)
        except ValueError:
            return


def _timed_out(job: Job, timeout_s: int) -> bool:
    if timeout_s <= 0:
        return False
    return utc_now_unix() - int(job.created_at) >= int(timeout_s)


async def _cancel_comfy_prompt(client: ComfyUIClient, prompt_id: str) -> None:
    cancel = getattr(client, "delete_queue_entries", None)
    if not callable(cancel):
        return
    try:
        await cancel([prompt_id])
    except Exception:
        logger.warning("failed to cancel queued ComfyUI prompt: prompt_id=%s", prompt_id, exc_info=True)


def is_staged(value: str) -> bool:
    return value.startswith(STAGED_PREFIX)


def staged_path(value: str) -> str:
    return value[len(STAGED_PREFIX) :]


async def upload_staged_values(
    *,
    job: Job,
    client: ComfyUIClient,
    input_subdir: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    values: list[str] = []
    if is_staged(job.image):
        values.append(job.image)
    for raw in job.standard_params.values():
        if isinstance(raw, str) and is_staged(raw):
            values.append(raw)
    for marker in values:
        if marker in mapping:
            continue
        path = staged_path(marker)
        data = Path(path).read_bytes()
        filename = Path(path).name
        mapping[marker] = await client.upload_image_bytes(
            data=data,
            filename=filename,
            subfolder=input_subdir,
            folder_type="input",
            overwrite=True,
        )
    return mapping


def _rewrite_params(params: dict[str, object], mapping: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, str) and value in mapping:
            out[key] = mapping[value]
        else:
            out[key] = value
    return out
