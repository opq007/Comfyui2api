from __future__ import annotations

import asyncio
import copy
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlencode

from .backend_store import BackendStore
from .comfy_client import ComfyApiError, ComfyUIClient, history_entry_error, history_entry_is_complete
from .comfy_workflow import (
    iter_file_outputs,
    normalize_prompt_enum_inputs,
    prepare_prompt,
    prune_invalid_orphan_output_nodes,
)
from .config import Config
from .instance_pool import InstancePool
from .job_scheduler import JobScheduler
from .signed_urls import create_signed_query, signing_secret
from .util import guess_media_type, json_dumps, pick_primary_url, sanitize_filename_part, utc_now_iso, utc_now_unix
from .workflow_params import apply_img2img_resize_params, normalize_standard_params, resolve_standard_overrides
from .workflow_registry import WorkflowDefinition, WorkflowRegistry


logger = logging.getLogger(__name__)


# Pull-based completion watchdog. The ComfyUI WebSocket is the fast path, but it
# is a fire-and-forget push channel: ComfyUI delivers `executing`/progress only
# to the socket registered under the prompt's client_id and silently drops the
# message when nobody is attached yet (the monitor connects *after* `/prompt`
# returns), a backend restart removes the prompt without a terminal event, and
# an operator emptying the queue does the same. Any of those leaves the gateway
# waiting on a socket that stays open and silent: the job sits in "queued" at 0%
# while its instance slot stays reserved until TIMEOUT_S (one hour by default),
# which reads exactly as "the task is stuck in the queue and never runs" and
# blocks that instance for every later job too.
#
# So the wait is sliced: once a slice passes without any push message, the slice
# ends with a `/history` probe (a dict lookup on the ComfyUI side). 5s is 10x
# lighter than the 0.5s poller this replaced, and a job whose WS is streaming
# pays nothing at all — silence after the job ends is what triggers the probe.
HISTORY_WATCH_INTERVAL_S = 5.0
# How long the push channel may stay quiet before the watchdog also probes
# `/queue`. Long enough that a normal sampler step (which streams WS events)
# never triggers it; the probe is what distinguishes "still pending on ComfyUI"
# from "the prompt is gone", and it is also how a lost `executing` event gets
# noticed so the task is not 0%/"queued" for its whole run.
WS_SILENCE_GRACE_S = 30.0
# A prompt that is neither executing nor pending on ComfyUI and has no history
# entry will never finish. Requiring several consecutive observations keeps a
# prompt that is merely moving from `queue_running` into history from being
# misread as lost.
PROMPT_MISSING_CHECKS = 3

# Where `/queue` says a prompt is. The values deliberately reuse the job status
# words ("running"/"pending"); the *names* are what keep the two domains apart —
# a queue state is not a job status and must never be assigned to one.
QueuePromptState = Literal["running", "pending", "missing", "unknown"]
QUEUE_STATE_RUNNING: QueuePromptState = "running"
QUEUE_STATE_PENDING: QueuePromptState = "pending"
QUEUE_STATE_MISSING: QueuePromptState = "missing"
# The backend could not answer, or answered in a shape we do not recognise. Never
# a reason to fail a job.
QUEUE_STATE_UNKNOWN: QueuePromptState = "unknown"

# ComfyUI events that belong to a prompt's execution. Everything else on the
# socket (`status`, node-pack telemetry) is broadcast to every client and says
# nothing about *this* job, so it must not count as activity.
PROMPT_EVENT_TYPES = frozenset(
    {
        "executing",
        "progress",
        "execution_start",
        "execution_cached",
        "execution_success",
        "execution_error",
        "executed",
    }
)


def _completion_timeout_error(timeout_s: int, prompt_id: str) -> TimeoutError:
    return TimeoutError(f"Timed out waiting for ComfyUI to finish after {timeout_s}s (prompt_id={prompt_id}).")


def _remaining_budget(timeout_s: int, deadline: float | None, now: float, prompt_id: str) -> int:
    """How much of TIMEOUT_S is still available, rounded up.

    Raises when nothing is left so a hand-over can never restart the clock.
    """
    if deadline is None:
        return timeout_s
    left_s = deadline - now
    if left_s <= 0:
        raise _completion_timeout_error(timeout_s, prompt_id)
    return max(1, int(left_s) + 1)


def _is_own_prompt_event(*, mtype: Any, pid: Any, prompt_id: str) -> bool:
    """Does this socket message belong to the job's own prompt?

    Only these may reset the watchdog's silence clock. `status` (and node-pack
    telemetry) is broadcast to every socket whenever the queue changes, so
    counting it would keep a stalled job looking alive as long as any other
    client touches the instance — the exact case the watchdog exists for.
    """
    if mtype not in PROMPT_EVENT_TYPES:
        return False
    if not pid:
        # Older ComfyUI omits prompt_id on job events; an unowned job event is
        # the best signal available there.
        return True
    return str(pid) == prompt_id


def _completed_history(entry: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """The entry when ComfyUI reports it final and successful, else None.

    A failed entry raises: that is a real ComfyUI outcome, not something to keep
    waiting on.
    """
    if not entry:
        return None
    error = history_entry_error(entry)
    if error:
        raise ComfyApiError(error)
    return entry if history_entry_is_complete(entry) else None


@dataclass
class _WsActivity:
    """Last time the monitor saw a message about this job's own prompt.

    Stamped with the event loop's clock (`loop.time()`) — the same clock the
    deadline uses, so the silence can be compared against it directly. Under
    uvloop `time.monotonic()` has a different origin and the subtraction would
    be meaningless.
    """

    last_event_at: float


def _signed_media_url(cfg: Config, url: str) -> str:
    """Sign a relative /runs/... URL so <img>/<video> tags can load it
    without an Authorization header.

    WS-pushed events (job_snapshot / job_completed) used to carry bare
    /runs/ paths, which the download endpoint rejects for browsers (they
    cannot set headers), so playground results showed as broken images.
    The signature covers path+expiry only, so it needs no request context,
    and consumers that re-sign (REST rewriters) strip sig/exp first —
    signing here is idempotent for them.
    """
    raw = str(url or "")
    if not raw.startswith("/runs/"):
        return raw
    secret = signing_secret(configured_secret=cfg.signed_url_secret, api_token=cfg.api_token)
    if not secret:
        return raw
    path = raw.split("?", 1)[0]
    signed = create_signed_query(path=path, ttl_seconds=cfg.signed_url_ttl_seconds, secret=secret)
    return f"{path}?{urlencode(signed)}"


def _request_image_count(params: Dict[str, Any], *, has_image: bool) -> int:
    """Reference image count for this request.

    AIO workflows select their branch from it. Prefer the count computed by
    `extra_edit_image_params` on multi-image paths, and fall back to the
    supplied primary image for endpoints that never call it.
    """
    raw = params.get("inputImgCount")
    if raw is not None and not isinstance(raw, bool):
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return 1 if has_image else 0


@dataclass
class JobOutput:
    filename: str
    url: str
    media_type: str
    node_id: str
    output_key: str


@dataclass
class Job:
    job_id: str
    created_at_utc: str
    created_at: int
    status: str
    kind: str
    workflow: str
    platform: str = "Native"
    requested_model: str = ""
    model_slug: str = ""
    instance_slug: str = ""
    seconds: str = ""
    size: str = ""
    quality: str = ""
    metadata: str = ""

    prompt: str = ""
    negative_prompt: str = ""
    image: str = ""

    prompt_node: str = ""
    negative_prompt_node: str = ""
    image_node: str = ""
    overrides: list[tuple[str, str, Any]] = field(default_factory=list)
    standard_params: Dict[str, Any] = field(default_factory=dict)

    prompt_id: str = ""
    client_id: str = ""
    queue_number: Optional[int] = None

    started_at_utc: str = ""
    finished_at_utc: str = ""
    updated_at_utc: str = ""
    duration_s: int | None = None

    current_node: str = ""
    progress: Dict[str, Any] = field(default_factory=dict)
    progress_percent: int = 0
    error: str = ""
    request_json: dict[str, Any] = field(default_factory=dict)

    run_dir: str = ""
    outputs: list[JobOutput] = field(default_factory=list)
    url: str = ""

    done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class JobManager:
    def __init__(
        self,
        *,
        cfg: Config,
        registry: WorkflowRegistry,
        pool: InstancePool,
        backend: BackendStore,
        store: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.pool = pool
        self.backend = backend
        self.store = store

        self._lock = asyncio.Lock()
        self._jobs: Dict[str, Job] = {}
        self.scheduler = JobScheduler(self)

        self._subscribers: Dict[str, set[Any]] = {}  # job_id -> set[WebSocket]
        self._sub_lock = asyncio.Lock()
        self._global_subscribers: set[Any] = set()
        self._global_sub_lock = asyncio.Lock()
        self._last_global_broadcast: float = 0.0

    # High-frequency telemetry that is broadcast at most once per
    # `global_broadcast_interval_s` to the global dashboard, instead of on every
    # sampler-step event. Terminal / status transitions are never throttled.
    _THROTTLED_GLOBAL_EVENTS = frozenset({"comfyui_ws", "job_progress"})

    async def start_workers(self) -> None:
        await self.scheduler.start()

    async def stop_workers(self) -> None:
        await self.scheduler.stop()

    async def create_job(
        self,
        *,
        kind: str,
        workflow: str,
        prompt: str,
        requested_model: str = "",
        seconds: str = "",
        size: str = "",
        quality: str = "",
        metadata: str = "",
        platform: str = "Native",
        negative_prompt: str = "",
        image: str = "",
        prompt_node: str = "",
        negative_prompt_node: str = "",
        image_node: str = "",
        overrides: Optional[list[tuple[str, str, Any]]] = None,
        standard_params: Optional[Dict[str, Any]] = None,
        request_json: Optional[dict[str, Any]] = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        params = dict(standard_params or {})
        if str(kind or "").strip().lower() == "img2img":
            params = apply_img2img_resize_params(params)
        # Every endpoint funnels through here, so standard params are validated
        # and coerced once, before enqueue: a bad value (e.g. speed outside the
        # sidecar's range) is a request-time InvalidParameterError instead of a
        # failed job. Explicit `overrides` stay raw node writes and are exempt.
        params = normalize_standard_params(spec=await self._parameter_spec(workflow), request_params=params)
        job = Job(
            job_id=job_id,
            created_at_utc=utc_now_iso(),
            created_at=utc_now_unix(),
            status="pending",
            kind=kind,
            workflow=workflow,
            platform=platform or "Native",
            requested_model=requested_model or "",
            model_slug=requested_model or "",
            instance_slug="",
            seconds=seconds or "",
            size=size or "",
            quality=quality or "",
            metadata=metadata or "",
            prompt=prompt or "",
            negative_prompt=negative_prompt or "",
            image=image or "",
            prompt_node=prompt_node or "",
            negative_prompt_node=negative_prompt_node or "",
            image_node=image_node or "",
            overrides=list(overrides or []),
            standard_params=params,
            updated_at_utc=utc_now_iso(),
            request_json=dict(request_json) if request_json is not None else self._default_request_summary(
                kind=kind,
                workflow=workflow,
                requested_model=requested_model,
                seconds=seconds,
                size=size,
                quality=quality,
                metadata=metadata,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                standard_params=params,
            ),
        )
        async with self._lock:
            self._jobs[job_id] = job
        await self._persist_job(job)
        self.scheduler.enqueue(job_id)
        await self._publish(job_id, {"type": "job_created", "data": self.public_job(job)})
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        requested = str(job_id or "").strip()
        if not requested:
            return None
        async with self._lock:
            live = self._jobs.get(requested)
            if live is not None:
                return live
        if self.store is None:
            return None
        payload = await self.store.get_task(requested)
        if payload is None:
            return None
        return job_from_stored_payload(payload)

    async def list_jobs(self, *, limit: int = 100) -> list[Job]:
        async with self._lock:
            return list(self._jobs.values())[-max(1, int(limit)) :]

    def public_job(self, job: Job) -> Dict[str, Any]:
        return {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "created_at_utc": job.created_at_utc,
            "started_at_utc": job.started_at_utc or None,
            "finished_at_utc": job.finished_at_utc or None,
            "updated_at_utc": job.updated_at_utc or None,
            "duration_s": job.duration_s,
            "status": job.status,
            "kind": job.kind,
            "workflow": job.workflow,
            "platform": job.platform,
            "requested_model": job.requested_model or job.model_slug or None,
            "model_slug": job.model_slug or job.requested_model or None,
            "instance_slug": job.instance_slug or None,
            "seconds": job.seconds or None,
            "size": job.size or None,
            "quality": job.quality or None,
            "metadata": job.metadata or None,
            "prompt_id": job.prompt_id or None,
            "queue_number": job.queue_number,
            "current_node": job.current_node or None,
            "progress": job.progress or None,
            "progress_percent": job.progress_percent,
            "error": job.error or None,
            "request_json": job.request_json or {},
            "url": _signed_media_url(self.cfg, job.url or "") or None,
            "output_count": len(job.outputs or []),
            "outputs": [self._public_output(o) for o in (job.outputs or [])],
        }

    def _public_output(self, output: JobOutput) -> Dict[str, Any]:
        payload = dict(output.__dict__)
        payload["url"] = _signed_media_url(self.cfg, output.url or "")
        return payload

    async def subscribe(self, job_id: str, ws: Any) -> None:
        async with self._sub_lock:
            self._subscribers.setdefault(job_id, set()).add(ws)

    async def unsubscribe(self, job_id: str, ws: Any) -> None:
        async with self._sub_lock:
            s = self._subscribers.get(job_id)
            if not s:
                return
            s.discard(ws)
            if not s:
                self._subscribers.pop(job_id, None)

    async def subscribe_all(self, ws: Any) -> None:
        async with self._global_sub_lock:
            self._global_subscribers.add(ws)

    async def unsubscribe_all(self, ws: Any) -> None:
        async with self._global_sub_lock:
            self._global_subscribers.discard(ws)

    async def _publish(self, job_id: str, event: Dict[str, Any]) -> None:
        payload = copy.deepcopy(event)
        async with self._sub_lock:
            sockets = list(self._subscribers.get(job_id, set()))
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.unsubscribe(job_id, ws)
        await self._publish_global_snapshot(job_id, event_type=str(event.get("type") or "task_updated"))

    async def _update(self, job_id: str, *, persist: bool = True, **fields: Any) -> Optional[Job]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            self._apply_derived_fields(job, fields)
            updated = job
        if persist:
            await self._persist_job(updated)
        if "outputs" in fields and self.store is not None:
            await self.store.replace_outputs(job_id, updated.outputs or [])
        return updated

    def _default_request_summary(
        self,
        *,
        kind: str,
        workflow: str,
        requested_model: str,
        seconds: str,
        size: str,
        quality: str,
        metadata: str,
        prompt: str,
        negative_prompt: str,
        image: str,
        standard_params: Dict[str, Any],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "kind": kind,
            "workflow": workflow,
            "model": requested_model or None,
            "prompt": _truncate_text(prompt),
            "negative_prompt": _truncate_text(negative_prompt),
            "seconds": seconds or None,
            "size": size or None,
            "quality": quality or None,
            "metadata": _truncate_text(metadata),
            "standard_params": dict(standard_params or {}),
        }
        if image:
            summary["image"] = {"source": "stored_input", "value": _truncate_text(image, limit=180)}
        return {k: v for k, v in summary.items() if v not in (None, "", {}, [])}

    def _apply_derived_fields(self, job: Job, changed: Dict[str, Any]) -> None:
        now_iso = utc_now_iso()
        job.updated_at_utc = str(changed.get("updated_at_utc") or now_iso)
        if job.status == "running" and not job.started_at_utc:
            job.started_at_utc = now_iso
        if "progress" in changed:
            job.progress_percent = progress_percent_from_progress(job.progress, status=job.status)
        if "status" in changed or "progress" in changed:
            job.progress_percent = progress_percent_from_progress(job.progress, status=job.status)
        if job.status in {"completed", "failed"}:
            if not job.finished_at_utc:
                job.finished_at_utc = now_iso
            job.progress_percent = 100
            job.duration_s = max(0, utc_now_unix() - int(job.created_at))

    async def _persist_job(self, job: Job) -> None:
        if self.store is None:
            return
        await self.store.upsert_job(job)

    async def _publish_global_snapshot(self, job_id: str, *, event_type: str) -> None:
        async with self._global_sub_lock:
            sockets = list(self._global_subscribers)
        if not sockets:
            return
        # High-frequency telemetry (per sampler step) is throttled to a single
        # aggregated broadcast within `global_broadcast_interval_s` instead of
        # re-serializing + sending the whole job to every dashboard per event.
        # Terminal / status transitions always go through immediately.
        if event_type in self._THROTTLED_GLOBAL_EVENTS:
            now = time.monotonic()
            async with self._lock:
                last = getattr(self, "_last_global_broadcast", 0.0)
                if now - last < self.cfg.global_broadcast_interval_s:
                    return
                self._last_global_broadcast = now
        job = await self.get_job(job_id)
        if job is None:
            return
        payload = {
            "type": "task_updated",
            "event": event_type,
            "ts": utc_now_iso(),
            "job": self.public_job(job),
        }
        for ws in sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.unsubscribe_all(ws)

    async def fail_job(self, job_id: str, error_message: str) -> None:
        job = await self.get_job(job_id)
        await self._update(job_id, status="failed", error=error_message)
        await self._publish(job_id, {"type": "job_failed", "data": {"error": error_message}})
        if job:
            job.done.set()

    async def _resolve_workflow(self, name: str) -> WorkflowDefinition:
        wf = await self.registry.get(name)
        if wf:
            return wf
        for item in await self.registry.list():
            if item.name.lower() == name.lower():
                return item
        raise FileNotFoundError(f"Workflow not found: {name}")

    async def _parameter_spec(self, workflow: str) -> Any:
        """Sidecar spec for a workflow name, or None when it cannot be resolved.

        Resolves exactly like `_run_job`, so validation and execution agree on
        which sidecar a job runs against. Test managers pass a bare registry
        stub; with nothing to resolve, the request-time check is skipped and
        the run-time resolution still normalizes what reached the job.
        """
        if not callable(getattr(self.registry, "get", None)):
            return None
        try:
            wf = await self._resolve_workflow(workflow)
        except FileNotFoundError:
            return None
        return wf.parameter_spec

    async def _run_job(self, job_id: str, *, client: ComfyUIClient) -> None:
        job = await self.get_job(job_id)
        if not job:
            return

        wf = await self._resolve_workflow(job.workflow)
        job_obj = wf.clone_obj()

        client_id = uuid.uuid4().hex
        await self._update(job_id, client_id=client_id, status="queued")
        await self._publish(job_id, {"type": "job_queued", "data": {"client_id": client_id, "workflow": wf.name}})

        logger.info(
            "job prompt request: %s",
            json_dumps(
                {
                    "job_id": job_id,
                    "workflow": wf.name,
                    "kind": job.kind,
                    "requested_model": job.requested_model or None,
                    "prompt": job.prompt or None,
                    "negative_prompt": job.negative_prompt or None,
                }
            ),
        )

        spec = wf.parameter_spec
        request_params = dict(job.standard_params or {})
        image_count = _request_image_count(request_params, has_image=bool(job.image))
        if spec is not None and "inputImgCount" in spec.parameters and "inputImgCount" not in request_params:
            # Chat and variations endpoints never call extra_edit_image_params,
            # but AIO workflows select their branch from this count. Derive it
            # from what was actually supplied instead of letting the sidecar
            # default pin the wrong branch.
            request_params["inputImgCount"] = image_count
        positive_prompt_node = job.prompt_node or (spec.prompt_node if spec is not None else "") or None
        # AIO workflows switch sampler *and* negative encoder on the image
        # count, so the sidecar supplies one negative target per branch.
        negative_prompt_node = job.negative_prompt_node or (
            spec.negative_prompt_node_for(image_count=image_count) if spec is not None else ""
        ) or None
        image_node = job.image_node or (spec.image_node if spec is not None else "") or None
        if spec is not None and "seed" in spec.parameters:
            raw_seed = request_params.get("seed")
            if raw_seed in (None, "", -1, "-1"):
                request_params["seed"] = random.randint(0, 2**32 - 1)

        prompt_graph, extra_data, applied, prompt_trace = prepare_prompt(
            workflow_obj=job_obj,
            positive_prompt=job.prompt or None,
            negative_prompt=job.negative_prompt or None,
            positive_prompt_node=positive_prompt_node,
            negative_prompt_node=negative_prompt_node,
            image=job.image or None,
            image_node=image_node,
            overrides=resolve_standard_overrides(
                workflow_obj=job_obj,
                spec=wf.parameter_spec,
                request_params=request_params,
            )
            + list(job.overrides or []),
        )
        object_info = await client.object_info()
        removed_nodes = prune_invalid_orphan_output_nodes(prompt_graph, object_info=object_info)
        normalized_inputs = normalize_prompt_enum_inputs(prompt_graph, object_info=object_info)
        if removed_nodes or normalized_inputs:
            logger.info(
                "sanitized prompt graph: job_id=%s removed_nodes=%s normalized_inputs=%s",
                job_id,
                removed_nodes,
                normalized_inputs,
            )

        logger.info(
            "job prompt prepared: %s",
            json_dumps(
                {
                    "job_id": job_id,
                    "workflow": wf.name,
                    "kind": job.kind,
                    "requested_model": job.requested_model or None,
                    "requested_prompt": job.prompt or None,
                    "requested_negative_prompt": job.negative_prompt or None,
                    "effective_positive_prompts": prompt_trace.get("positive") or [],
                    "effective_negative_prompts": prompt_trace.get("negative") or [],
                }
            ),
        )

        qp = await client.queue_prompt(prompt=prompt_graph, client_id=client_id, extra_data=extra_data)
        await self._update(job_id, prompt_id=qp.prompt_id, queue_number=qp.number)
        await self._publish(
            job_id,
            {
                "type": "comfyui_prompt_queued",
                "data": {"prompt_id": qp.prompt_id, "queue_number": qp.number, "overrides": applied},
            },
        )

        run_dir = (self.cfg.runs_dir / job_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "prompt.json").write_text(
            json_dumps({"prompt": prompt_graph, "extra_data": extra_data}) + "\n", encoding="utf-8"
        )
        await self._update(job_id, run_dir=str(run_dir))

        # The WebSocket monitor is the live, push-based progress source. It is
        # not the only completion source, though: `_await_completion` races it
        # against a pull-based watchdog, so a lost push costs one cheap
        # `/history` probe instead of a job parked in "queued" (holding its
        # instance slot) until TIMEOUT_S.
        activity = _WsActivity(last_event_at=asyncio.get_running_loop().time())
        ws_task = asyncio.create_task(
            self._monitor_ws(
                job_id=job_id,
                client=client,
                client_id=client_id,
                prompt_id=qp.prompt_id,
                activity=activity,
            ),
            name=f"job-ws-{job_id[:8]}",
        )
        try:
            history = await self._await_completion(
                job_id=job_id,
                client=client,
                prompt_id=qp.prompt_id,
                ws_task=ws_task,
                activity=activity,
            )
        finally:
            # Cancellation (shutdown / scheduler stop) must propagate without
            # stalling in a history poll, and a finished wait must never leave
            # the monitor connected until its ping timeout.
            if not ws_task.done():
                ws_task.cancel()
            await asyncio.gather(ws_task, return_exceptions=True)

        (run_dir / "history.json").write_text(json_dumps(history) + "\n", encoding="utf-8")

        outputs: list[JobOutput] = []
        # iter_file_outputs yields only the entries a save node wrote to the
        # output folder, with filename/subfolder/type already normalized.
        for node_id, output_key, fileinfo in iter_file_outputs(history):
            filename = fileinfo["filename"]
            blob = await client.view_bytes(
                filename=filename,
                subfolder=fileinfo["subfolder"],
                folder_type=fileinfo["type"],
            )
            safe_node = sanitize_filename_part(node_id, max_len=60)
            safe_name = sanitize_filename_part(Path(filename).name, max_len=120)
            out_name = f"{safe_node}__{safe_name}"
            out_path = run_dir / out_name
            out_path.write_bytes(blob)

            url = f"/runs/{job_id}/{out_name}"
            outputs.append(
                JobOutput(
                    filename=out_name,
                    url=url,
                    media_type=guess_media_type(out_name),
                    node_id=node_id,
                    output_key=output_key,
                )
            )

        primary = pick_primary_url([{"filename": o.filename, "url": o.url} for o in outputs]) if outputs else None
        await self._update(job_id, status="completed", outputs=outputs, url=primary or "")
        await self._publish(
            job_id,
            {
                "type": "job_completed",
                "data": {
                    # Signed so browser <img>/<video> tags on the playground can
                    # load the media straight from the WS event payload.
                    "url": _signed_media_url(self.cfg, primary or "") or None,
                    "outputs": [self._public_output(o) for o in outputs],
                },
            },
        )

        job = await self.get_job(job_id)
        if job:
            job.done.set()

    async def _await_completion(
        self,
        *,
        job_id: str,
        client: ComfyUIClient,
        prompt_id: str,
        ws_task: asyncio.Task[None],
        activity: _WsActivity,
    ) -> Dict[str, Any]:
        """Wait for the job's ComfyUI prompt to finish.

        The WebSocket monitor is awaited in bounded slices so a pull-based
        watchdog can run alongside it. A slice that saw no push message ends
        with one `/history` probe (a dict lookup on ComfyUI's side), because
        silence is not proof the prompt is alive: a prompt that finished while
        the push channel was quiet is still collected here instead of leaving
        the job — and its instance slot — in "queued" for the whole TIMEOUT_S.
        While WS events keep flowing the probe never runs, so the "no redundant
        poller" cost model that removed the old 0.5s poller still holds. A job
        whose socket stays silent for its whole run pays at most one `/history`
        and one `/queue` per `HISTORY_WATCH_INTERVAL_S`.

        The `/queue` probe behind WS_SILENCE_GRACE_S distinguishes a prompt still
        waiting its turn from one ComfyUI no longer knows about (backend
        restarted, queue cleared). It also repairs a lost `executing` event by
        reporting the task as running instead of 0%/"queued".

        Only the execution window is bounded here. The queue wait is bounded by
        the scheduler from `created_at` (spec item 12), and these are two
        separate budgets by design: a task that waited a long time for a free
        slot must not be killed the moment it starts running.
        """
        loop = asyncio.get_running_loop()
        timeout_s = self.cfg.timeout_s
        poll_interval_s = float(self.cfg.poll_interval_s)
        deadline = None if timeout_s <= 0 else loop.time() + float(timeout_s)
        watch_interval = max(HISTORY_WATCH_INTERVAL_S, poll_interval_s)
        silence_grace = WS_SILENCE_GRACE_S
        if timeout_s > 0:
            # The watchdog has to be able to act before the deadline; a short
            # TIMEOUT_S must not turn the grace period into "never probes".
            silence_grace = min(silence_grace, max(watch_interval, float(timeout_s) / 10.0))
        missing_checks = 0
        while True:
            remaining = watch_interval if deadline is None else min(watch_interval, deadline - loop.time())
            if remaining <= 0:
                raise _completion_timeout_error(timeout_s, prompt_id)
            done, _pending = await asyncio.wait({ws_task}, timeout=remaining)
            if ws_task in done:
                return await self._history_after_ws_end(
                    job_id=job_id,
                    client=client,
                    prompt_id=prompt_id,
                    ws_task=ws_task,
                    # Only the budget actually left: a backend that stalls twice
                    # must not cost 2x TIMEOUT_S.
                    timeout_s=_remaining_budget(timeout_s, deadline, loop.time(), prompt_id),
                    poll_interval_s=poll_interval_s,
                )
            silent_for = loop.time() - activity.last_event_at
            if silent_for < watch_interval:
                # Events are still arriving; the push channel is doing its job.
                missing_checks = 0
                continue
            history, missing_checks = await self._watchdog_probe(
                job_id=job_id,
                client=client,
                prompt_id=prompt_id,
                missing_checks=missing_checks,
                check_queue=silent_for >= silence_grace,
            )
            if history is not None:
                # The prompt finished while the push channel was silent: this is
                # the write-back the job would otherwise never get.
                return history

    async def _watchdog_probe(
        self,
        *,
        job_id: str,
        client: ComfyUIClient,
        prompt_id: str,
        missing_checks: int,
        check_queue: bool,
    ) -> tuple[Dict[str, Any] | None, int]:
        """One pull probe while the push channel is quiet.

        Returns (finished history, updated missing-prompt streak); a non-None
        history ends the wait. Failing a job for a vanished prompt requires both
        probes to be readable, so an unreadable `/history` is never evidence.
        """
        history_ok, entry = await _poll_history_entry(client, prompt_id)
        finished = _completed_history(entry)
        if finished is not None:
            return finished, missing_checks
        if not check_queue:
            return None, 0
        state = await _queue_prompt_state(client, prompt_id)
        if state == QUEUE_STATE_RUNNING:
            await self._report_running_from_watchdog(job_id)
            return None, 0
        if state != QUEUE_STATE_MISSING or not history_ok:
            # Still pending on ComfyUI, an unusable probe result, or a `/queue`
            # miss we cannot corroborate because history is unreadable: keep
            # waiting. Failing on partial evidence would throw away the outputs
            # of a prompt that actually finished.
            return None, 0
        missing_checks += 1
        if missing_checks >= PROMPT_MISSING_CHECKS:
            # ComfyUI has neither queued nor run this prompt: waiting out
            # TIMEOUT_S would only hold the instance slot, so fail with the real
            # reason. The scheduler sees `prompt_id` and fails the job without
            # rerouting, per spec §11.6 (never re-spend the GPU on a prompt that
            # may still be alive somewhere).
            raise ComfyApiError(
                f"ComfyUI no longer has this prompt queued or in its history (prompt_id={prompt_id})."
            )
        return None, missing_checks

    async def _report_running_from_watchdog(self, job_id: str) -> None:
        """Mark a job running when its `executing` event was lost.

        The node id lives in the event we never received, so it is reported as
        unknown rather than invented.
        """
        job = await self.get_job(job_id)
        if job is None or job.status == "running":
            return
        await self._update(job_id, status="running", persist=True)
        await self._publish(
            job_id,
            {"type": "job_running", "data": {"node": None, "instance_slug": job.instance_slug or None}},
        )

    async def _history_after_ws_end(
        self,
        *,
        job_id: str,
        client: ComfyUIClient,
        prompt_id: str,
        ws_task: asyncio.Task[None],
        timeout_s: int,
        poll_interval_s: float,
    ) -> Dict[str, Any]:
        """Collect the history after the monitor finished on its own."""
        ws_error: BaseException | None = None
        if not ws_task.cancelled():
            ws_error = ws_task.exception()
        if ws_error is not None:
            logger.warning(
                "job websocket monitor failed, continuing with history polling: job_id=%s prompt_id=%s exc=%r",
                job_id,
                prompt_id,
                ws_error,
            )
            return await client.wait_for_history_complete(
                prompt_id=prompt_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s
            )
        entry = await client.get_history_entry(prompt_id)
        finished = _completed_history(entry)
        if finished is not None:
            return finished
        # WS reported completion but history is not final yet (or the WS ended
        # without a terminal signal). Fall back to bounded history polling.
        return await client.wait_for_history_complete(
            prompt_id=prompt_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s
        )

    async def _monitor_ws(
        self,
        *,
        job_id: str,
        client: ComfyUIClient,
        client_id: str,
        prompt_id: str,
        activity: _WsActivity,
    ) -> None:
        async for msg in client.ws_events(client_id=client_id):
            mtype = msg.get("type")
            data = msg.get("data", {})
            if _is_own_prompt_event(
                mtype=mtype,
                pid=data.get("prompt_id") if isinstance(data, dict) else None,
                prompt_id=prompt_id,
            ):
                # Only this job's own events keep the watchdog thinking the push
                # channel is alive; foreign broadcasts must not mask a stall.
                activity.last_event_at = asyncio.get_running_loop().time()
            await self._publish(job_id, {"type": "comfyui_ws", "data": msg})
            if mtype == "executing" and isinstance(data, dict):
                pid = data.get("prompt_id")
                node = data.get("node")
                if pid and pid != prompt_id:
                    continue
                if node is None:
                    return
                # Persist the transitioning node id (rare), but do not write
                # SQLite for every progress message.
                updated = await self._update(job_id, status="running", current_node=str(node), persist=True)
                await self._publish(
                    job_id,
                    {
                        "type": "job_running",
                        "data": {"node": str(node), "instance_slug": (updated.instance_slug if updated else None) or None},
                    },
                )
            elif mtype == "execution_success" and isinstance(data, dict):
                pid = data.get("prompt_id")
                if pid and pid != prompt_id:
                    continue
                return
            elif mtype == "progress" and isinstance(data, dict):
                # Progress updates are high-frequency (per sampler step). Update
                # only the in-memory job + WS push; defer the SQLite write to a
                # terminal/status transition. This removes a full-row upsert +
                # new sqlite connection per progress event.
                await self._update(job_id, progress=data, persist=False)
                await self._publish(job_id, {"type": "job_progress", "data": data})
            elif mtype == "execution_error" and isinstance(data, dict):
                pid = data.get("prompt_id")
                if pid and pid != prompt_id:
                    continue
                raise ComfyApiError(str(data))


def job_from_stored_payload(payload: dict[str, Any]) -> Job:
    raw_task = payload.get("task")
    task: dict[str, Any] = raw_task if isinstance(raw_task, dict) else {}
    job_id = str(task.get("job_id") or "")
    outputs: list[JobOutput] = []
    for item in payload.get("outputs") or []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename") or "").strip()
        if not filename:
            raw_url = str(item.get("url") or "").strip()
            filename = Path(raw_url).name if raw_url else ""
        if not filename:
            continue
        outputs.append(
            JobOutput(
                filename=filename,
                url=str(item.get("url") or f"/runs/{job_id}/{filename}"),
                media_type=str(item.get("media_type") or ""),
                node_id=str(item.get("node_id") or ""),
                output_key=str(item.get("output_key") or ""),
            )
        )

    job = Job(
        job_id=job_id,
        created_at_utc=str(task.get("created_at_utc") or ""),
        created_at=int(task.get("created_at") or 0),
        status=str(task.get("status") or ""),
        kind=str(task.get("kind") or ""),
        workflow=str(task.get("workflow") or ""),
        platform=str(task.get("platform") or "Native"),
        requested_model=str(task.get("requested_model") or task.get("model_slug") or ""),
        model_slug=str(task.get("model_slug") or task.get("requested_model") or ""),
        instance_slug=str(task.get("instance_slug") or ""),
        prompt=str(task.get("prompt_preview") or ""),
        prompt_id=str(task.get("prompt_id") or ""),
        queue_number=task.get("queue_number"),
        started_at_utc=str(task.get("started_at_utc") or ""),
        finished_at_utc=str(task.get("finished_at_utc") or ""),
        updated_at_utc=str(task.get("updated_at_utc") or ""),
        duration_s=task.get("duration_s"),
        current_node=str(task.get("current_node") or ""),
        progress=dict(task.get("progress") or {}) if isinstance(task.get("progress"), dict) else {},
        progress_percent=int(task.get("progress_percent") or 0),
        error=str(task.get("error") or ""),
        request_json=dict(task.get("request_json") or {}) if isinstance(task.get("request_json"), dict) else {},
        outputs=outputs,
        url=str(task.get("url") or ""),
    )
    if job.status in {"completed", "failed"}:
        job.done.set()
    return job


def _truncate_text(value: str, *, limit: int = 500) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


async def _poll_history_entry(client: ComfyUIClient, prompt_id: str) -> tuple[bool, Dict[str, Any] | None]:
    """(readable, entry) for one history probe.

    "Cannot read" is reported separately from "not finished yet" because the
    caller fails jobs on absence evidence: a broken `/history` must never be
    mistaken for a prompt that left no trace.
    """
    try:
        return True, await client.get_history_entry(prompt_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("history watchdog probe failed: prompt_id=%s error=%r", prompt_id, exc)
        return False, None


async def _queue_prompt_state(client: ComfyUIClient, prompt_id: str) -> QueuePromptState:
    """Where a prompt currently sits on ComfyUI.

    Returns QUEUE_STATE_UNKNOWN whenever the backend cannot answer or answers in
    a shape we do not recognise, so a flaky probe can never fail a healthy job.
    """
    try:
        payload = await client.get_queue()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("queue watchdog probe failed: prompt_id=%s error=%r", prompt_id, exc)
        return QUEUE_STATE_UNKNOWN
    if not isinstance(payload, dict):
        return QUEUE_STATE_UNKNOWN
    recognized = 0
    for key, state in (("queue_running", QUEUE_STATE_RUNNING), ("queue_pending", QUEUE_STATE_PENDING)):
        entries = payload.get(key)
        if not isinstance(entries, list):
            continue
        recognized += 1
        for entry in entries:
            if isinstance(entry, (list, tuple)) and len(entry) > 1:
                if str(entry[1]) == prompt_id:
                    return state
            elif isinstance(entry, dict) and str(entry.get("prompt_id") or "") == prompt_id:
                return state
    if not recognized:
        # Unrecognised shape (another ComfyUI flavour / a proxy in front): never
        # read that as "the prompt is gone".
        return QUEUE_STATE_UNKNOWN
    return QUEUE_STATE_MISSING


def progress_percent_from_progress(progress: Dict[str, Any], *, status: str) -> int:
    if status in {"completed", "failed"}:
        return 100
    if status in {"pending", "queued"}:
        return 0
    if not isinstance(progress, dict):
        return 0
    try:
        value = float(progress.get("value") or 0)
        total = float(progress.get("max") or 0)
    except Exception:
        return 0
    if total <= 0:
        return 0
    pct = int((value / total) * 100.0)
    return max(0, min(99, pct))
