from __future__ import annotations

import asyncio
import ipaddress
import logging
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from .admin_limiter import AdminAuthLimiter
from .backend_store import BackendStore
from .comfy_workflow import (
    extract_prompt_and_extra,
    find_load_image_targets,
    find_text_prompt_targets,
    pick_unique_load_image_target,
    pick_unique_target,
)
from .config import Config
from .errors import ConflictError, DuplicateError, NotFoundError, SlugError, ValidationError
from .instance_pool import InstancePool
from .job_store import JobStore
from .jobs import JobManager
from .model_catalog import ModelCatalog, capability_kinds
from .signed_urls import create_signed_query, signing_secret
from .util import bearer_authorized
from .workflow_params import detect_parameter_candidates, generate_parameter_template, public_parameter_spec


logger = logging.getLogger(__name__)


def create_admin_router() -> APIRouter:
    router = APIRouter(prefix="/v1/admin", tags=["admin"])

    @router.get("/tasks")
    async def list_tasks(
        request: Request,
        start: str | None = None,
        end: str | None = None,
        q: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        platform: str | None = None,
        has_outputs: bool = Query(False),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        cfg = _cfg(request)
        payload = await _store(request).list_tasks(
            start=start,
            end=end,
            q=q,
            statuses=_split_csv(status),
            kinds=_split_csv(kind),
            platforms=_split_csv(platform),
            has_outputs=has_outputs,
            limit=limit,
            offset=offset,
        )
        items = []
        for raw in payload.get("items") or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            job_id = str(item.get("job_id") or "")
            raw_primary = str(item.get("url") or "")
            primary_name = Path(raw_primary).name if raw_primary else ""
            if primary_name and job_id:
                item["url"] = _authorized_url(request, cfg, f"/runs/{job_id}/{primary_name}")
            items.append(item)
        payload["items"] = items
        return payload

    @router.get("/tasks/{job_id}")
    async def get_task(
        request: Request,
        job_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        cfg = _cfg(request)
        _require_admin_auth(request, authorization)
        payload = await _store(request).get_task(job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Task not found"}})
        return _rewrite_task_payload_urls(request, cfg, payload)

    @router.get("/stats")
    async def stats(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        cfg = _cfg(request)
        _require_admin_auth(request, authorization)
        pool: InstancePool = request.app.state.pool
        instances = await _backend(request).list_instances()
        healthy = 0
        for item in instances:
            snap = await pool.runtime_snapshot(item.slug)
            if snap.get("health") == "healthy":
                healthy += 1
        base = await _store(request).stats()
        base.update(
            {
                "instance_count": len(instances),
                "healthy_instance_count": healthy,
                "workflows_dir": str(cfg.workflows_dir),
                "runs_dir": str(cfg.runs_dir),
                "database_path": str(cfg.database_path),
                "ui_enabled": cfg.ui_enabled,
            }
        )
        return base

    @router.get("/playground/submissions")
    async def list_playground_submissions(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        cfg = _cfg(request)
        _require_admin_auth(request, authorization)
        payload = await _store(request).list_submissions(limit=limit, offset=offset)
        items = []
        for raw in payload.get("items") or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            # Result URLs are relative /runs/... paths; sign them so dashboard
            # <img>/<video> tags can load the media without auth headers.
            raw_result = str(item.get("result_url") or "")
            if raw_result:
                item["result_url"] = _authorized_url(request, cfg, raw_result)
            items.append(item)
        payload["items"] = items
        return payload

    @router.post("/playground/submissions")
    async def add_playground_submission(
        request: Request,
        body: dict[str, Any],
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        if not str(body.get("job_id") or "").strip():
            raise _admin_error(400, "job_id is required")
        record = await _store(request).add_submission(
            kind=str(body.get("kind") or ""),
            model=str(body.get("model") or ""),
            params=dict(body.get("params") or {}) if isinstance(body.get("params"), dict) else {},
            job_id=str(body.get("job_id") or ""),
        )
        raw_result = str(record.get("result_url") or "")
        if raw_result:
            record["result_url"] = _authorized_url(request, _cfg(request), raw_result)
        return record

    @router.delete("/playground/submissions")
    async def clear_playground_submissions(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        return await _store(request).clear_submissions()

    @router.get("/workflows")
    async def workflows(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        cfg = _cfg(request)
        _require_admin_auth(request, authorization)
        return {"workflows_dir": str(cfg.workflows_dir), "items": await _workflow_items(request)}

    @router.get("/workflows/{name}/targets")
    async def workflow_targets(request: Request, name: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        wf = await _resolve_workflow_name(request, name)
        prompt, _extra = extract_prompt_and_extra(wf.workflow_obj)
        pos, neg = find_text_prompt_targets(prompt)
        img = find_load_image_targets(prompt)

        def _as_candidates(items: list[tuple[str, str, str, str]]) -> list[dict[str, Any]]:
            return [
                {
                    "ref": f"{node_id}.{input_key}",
                    "node_id": node_id,
                    "input_key": input_key,
                    "class_type": cls,
                    "title": title or None,
                }
                for node_id, input_key, cls, title in items
            ]

        def _try_pick_text(kind: str, candidates: list[tuple[str, str, str, str]]) -> tuple[str | None, str | None]:
            try:
                node_id, input_key = pick_unique_target(kind=kind, candidates=candidates)
                return f"{node_id}.{input_key}", None
            except Exception as exc:
                return None, str(exc)

        def _try_pick_image(candidates: list[tuple[str, str, str, str]]) -> tuple[str | None, str | None]:
            try:
                node_id, input_key = pick_unique_load_image_target(candidates)
                return f"{node_id}.{input_key}", None
            except Exception as exc:
                return None, str(exc)

        pos_auto, pos_err = _try_pick_text("positive", pos)
        neg_auto, neg_err = _try_pick_text("negative", neg)
        img_auto, img_err = _try_pick_image(img)
        return {
            "workflow": {"name": wf.name, "kind": wf.capabilities.kind, "mtime_ns": wf.mtime_ns},
            "targets": {
                "positive_prompt": {"autodetect": pos_auto, "autodetect_error": pos_err, "candidates": _as_candidates(pos)},
                "negative_prompt": {"autodetect": neg_auto, "autodetect_error": neg_err, "candidates": _as_candidates(neg)},
                "image": {"autodetect": img_auto, "autodetect_error": img_err, "candidates": _as_candidates(img)},
            },
        }

    @router.get("/workflows/{name}/parameters")
    async def workflow_parameters(request: Request, name: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        wf = await _resolve_workflow_name(request, name)
        return {
            "workflow": {"name": wf.name, "kind": wf.capabilities.kind, "mtime_ns": wf.mtime_ns},
            "parameter_mapping": public_parameter_spec(wf.parameter_spec),
            "detected_candidates": detect_parameter_candidates(wf.workflow_obj),
            "suggested_template": generate_parameter_template(
                workflow_obj=wf.workflow_obj, kind=wf.capabilities.kind, spec=wf.parameter_spec
            ),
            "parameter_error": wf.parameter_error,
        }

    @router.get("/workflows/{name}/parameters/template")
    async def workflow_parameters_template(
        request: Request, name: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        wf = await _resolve_workflow_name(request, name)
        return {
            "workflow": {"name": wf.name, "kind": wf.capabilities.kind, "mtime_ns": wf.mtime_ns},
            "template": generate_parameter_template(
                workflow_obj=wf.workflow_obj, kind=wf.capabilities.kind, spec=wf.parameter_spec
            ),
            "parameter_error": wf.parameter_error,
        }

    @router.get("/instances")
    async def list_instances(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        return {"items": await _instance_payloads(request)}

    @router.post("/instances")
    async def create_instance(request: Request, body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        try:
            record = await _backend(request).create_instance(body)
        except (SlugError, ValidationError, DuplicateError) as exc:
            raise _admin_error(400 if not isinstance(exc, DuplicateError) else 409, str(exc)) from exc
        await request.app.state.pool.sync_instance(record, probe_now=record.enabled)
        return await _instance_payload(request, record.slug)

    @router.get("/instances/{slug}")
    async def get_instance(request: Request, slug: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        payload = await _instance_payload(request, slug)
        if payload is None:
            raise _admin_error(404, f"Instance `{slug}` not found.")
        return payload

    @router.patch("/instances/{slug}")
    async def patch_instance(
        request: Request, slug: str, body: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        try:
            record = await _backend(request).patch_instance(slug, body)
        except NotFoundError as exc:
            raise _admin_error(404, str(exc)) from exc
        except ConflictError as exc:
            raise _admin_error(409, str(exc)) from exc
        except DuplicateError as exc:
            raise _admin_error(409, str(exc)) from exc
        except (SlugError, ValidationError) as exc:
            raise _admin_error(400, str(exc)) from exc
        await request.app.state.pool.sync_instance(record, probe_now=record.enabled)
        return await _instance_payload(request, record.slug)

    @router.delete("/instances/{slug}")
    async def delete_instance(request: Request, slug: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
        _require_admin_auth(request, authorization)
        try:
            await _backend(request).delete_instance(slug)
        except NotFoundError as exc:
            raise _admin_error(404, str(exc)) from exc
        except ConflictError as exc:
            raise _admin_error(409, str(exc)) from exc
        await request.app.state.pool.drop_instance(slug)
        return {"status": "deleted"}

    @router.get("/models")
    async def list_models(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        items = []
        for record in await _backend(request).list_models():
            items.append(await _model_payload(request, record.slug))
        return {"items": items}

    @router.post("/models")
    async def create_model(request: Request, body: dict[str, Any], authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        payload = dict(body)
        if payload.get("enabled"):
            await _assert_enable(request, payload.get("workflow_name"))
        try:
            record = await _backend(request).create_model(payload)
        except (SlugError, ValidationError) as exc:
            raise _admin_error(400, str(exc)) from exc
        except DuplicateError as exc:
            raise _admin_error(409, str(exc)) from exc
        return await _model_payload(request, record.slug)

    @router.get("/models/{slug}")
    async def get_model(request: Request, slug: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        payload = await _model_payload(request, slug)
        if payload is None:
            raise _admin_error(404, f"Model `{slug}` not found.")
        return payload

    @router.patch("/models/{slug}")
    async def patch_model(
        request: Request, slug: str, body: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        _require_admin_auth(request, authorization)
        current = await _backend(request).get_model(slug)
        if current is None:
            raise _admin_error(404, f"Model `{slug}` not found.")
        payload = dict(body)
        enabling = bool(payload.get("enabled")) if "enabled" in payload else current.enabled
        workflow_name = payload.get("workflow_name") if "workflow_name" in payload else current.workflow_name
        if enabling:
            await _assert_enable(request, workflow_name)
        try:
            record = await _backend(request).patch_model(slug, payload)
        except NotFoundError as exc:
            raise _admin_error(404, str(exc)) from exc
        except ConflictError as exc:
            raise _admin_error(409, str(exc)) from exc
        except (SlugError, ValidationError) as exc:
            raise _admin_error(400, str(exc)) from exc
        return await _model_payload(request, record.slug)

    @router.delete("/models/{slug}")
    async def delete_model(request: Request, slug: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
        _require_admin_auth(request, authorization)
        try:
            await _backend(request).delete_model(slug)
        except NotFoundError as exc:
            raise _admin_error(404, str(exc)) from exc
        except ConflictError as exc:
            raise _admin_error(409, str(exc)) from exc
        return {"status": "deleted"}

    @router.post("/shutdown")
    async def shutdown(request: Request, authorization: str | None = Header(default=None)) -> dict[str, str]:
        _require_admin_auth(request, authorization)
        _require_local_request(request)
        callback = getattr(request.app.state, "shutdown_callback", None)
        if not callable(callback):
            raise HTTPException(status_code=409, detail={"error": {"message": "Shutdown is not available for this process"}})
        asyncio.get_running_loop().call_later(0.2, callback)
        return {"status": "shutting_down"}

    @router.websocket("/tasks/ws")
    async def tasks_ws(ws: WebSocket) -> None:
        try:
            _require_admin_auth_ws(ws)
        except HTTPException as exc:
            # Accept first, then send a structured JSON error frame and close.
            # Accepting avoids uvicorn mapping an unaccepted WebSocket close
            # (close code 1008) to an HTTP 403 response, which previously made
            # legitimate auth failures look like transport errors.
            await ws.accept()
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "data": {"status": exc.status_code, "message": str(exc.detail)},
                    }
                )
            except Exception:
                pass
            await ws.close(
                code=1008 if exc.status_code != 429 else 1013,
                reason=str(exc.detail),
            )
            return
        jobs: JobManager = ws.app.state.jobs
        store: JobStore = ws.app.state.job_store
        await ws.accept()
        await jobs.subscribe_all(ws)
        try:
            snapshot = await store.list_tasks(limit=50)
            await ws.send_json({"type": "snapshot", "data": snapshot})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await jobs.unsubscribe_all(ws)

    return router


def _cfg(request: Request) -> Config:
    return request.app.state.cfg


def _store(request: Request) -> JobStore:
    return request.app.state.job_store


def _backend(request: Request) -> BackendStore:
    return request.app.state.backend


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _client_ip(request: Request | WebSocket) -> str:
    client = getattr(request, "client", None)
    return (client.host if client else "") or ""


def _limiter(target: Request | WebSocket) -> AdminAuthLimiter:
    return target.app.state.admin_limiter


def _require_admin_auth(request: Request, authorization: str | None) -> None:
    _enforce_admin(request.app.state.cfg, _limiter(request), _client_ip(request), authorization)


def _require_admin_auth_ws(ws: WebSocket) -> None:
    _enforce_admin(ws.app.state.cfg, _limiter(ws), _client_ip(ws), _auth_value_from_ws(ws))


def _enforce_admin(cfg: Config, limiter: AdminAuthLimiter, ip: str, authorization: str | None) -> None:
    if limiter.is_blocked(ip):
        raise HTTPException(status_code=429, detail={"error": {"message": "Too many failed admin auth attempts"}})
    if not cfg.admin_token:
        # Server-side misconfiguration: ADMIN_TOKEN was not provided. Log it so
        # operators can spot the gap, but return the same opaque "Unauthorized"
        # to clients (no need to leak that the server has no token configured).
        logger.warning("admin auth rejected: ADMIN_TOKEN is not configured on the server")
        raise HTTPException(status_code=401, detail={"error": {"message": "Unauthorized"}})
    if not bearer_authorized(authorization or "", cfg.admin_token):
        limiter.note_failure(ip)
        # Log enough for the operator to see why auth failed without leaking the
        # configured token. The Authorization header is logged at DEBUG level
        # only (with redaction) so logs stay safe in shared environments.
        logger.info(
            "admin auth rejected for ip=%s (authorization header present: %s, length: %s)",
            ip,
            bool((authorization or "").strip()),
            len((authorization or "").strip()),
        )
        raise HTTPException(status_code=401, detail={"error": {"message": "Unauthorized"}})
    limiter.note_success(ip)


def _auth_value_from_query_params(query_params: Mapping[str, Any]) -> str | None:
    for key in ("authorization", "api_key", "token", "access_token"):
        raw_value = query_params.get(key)
        raw = str(raw_value or "").strip()
        if not raw:
            continue
        if key == "authorization" or raw.lower().startswith("bearer "):
            return raw
        return f"Bearer {raw}"
    return None


def _auth_value_from_ws(ws: WebSocket) -> str | None:
    header_value = (ws.headers.get("authorization") or "").strip()
    if header_value:
        return header_value
    return _auth_value_from_query_params(ws.query_params)


def _require_local_request(request: Request) -> None:
    host = (request.client.host if request.client else "") or ""
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        if host.lower() == "localhost":
            return
    raise HTTPException(status_code=403, detail={"error": {"message": "Shutdown is only available from localhost"}})


def _admin_error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"message": message}})


async def _assert_enable(request: Request, workflow_name: Any) -> None:
    catalog: ModelCatalog = request.app.state.catalog
    try:
        await catalog.assert_enableable(str(workflow_name) if workflow_name is not None else None)
    except ValidationError as exc:
        raise _admin_error(400, str(exc)) from exc


async def _instance_payloads(request: Request) -> list[dict[str, Any]]:
    items = []
    for record in await _backend(request).list_instances():
        payload = await _instance_payload(request, record.slug)
        if payload is not None:
            items.append(payload)
    return items


async def _instance_payload(request: Request, slug: str) -> dict[str, Any] | None:
    record = await _backend(request).get_instance(slug)
    if record is None:
        return None
    snap = await request.app.state.pool.runtime_snapshot(slug)
    bound = await _backend(request).bound_model_count(slug)
    payload = record.public_dict()
    payload.update(snap)
    payload["bound_model_count"] = bound
    return payload


async def _model_payload(request: Request, slug: str) -> dict[str, Any] | None:
    record = await _backend(request).get_model(slug)
    if record is None:
        return None
    catalog: ModelCatalog = request.app.state.catalog
    workflow, kinds, available = await catalog.workflow_meta(record.workflow_name)
    healthy = await request.app.state.pool.healthy_slugs(list(record.instance_slugs))
    payload = record.public_dict()
    payload["kind"] = list(kinds)
    payload["workflow_available"] = available
    payload["ready"] = available and bool(healthy)
    payload["workflow_kind"] = workflow.capabilities.kind if workflow is not None else None
    return payload


async def _workflow_items(request: Request) -> list[dict[str, Any]]:
    registry = request.app.state.registry
    items: list[dict[str, Any]] = []
    for wf in await registry.list():
        items.append(
            {
                "name": wf.name,
                "kind": wf.capabilities.kind,
                "kinds": list(capability_kinds(wf.capabilities)),
                "available": True,
                "load_error": None,
                "parameter_error": wf.parameter_error,
            }
        )
    for load_error in await registry.list_load_errors():
        items.append(
            {
                "name": load_error.name,
                "kind": None,
                "kinds": [],
                "available": False,
                "load_error": load_error.error,
                "parameter_error": None,
            }
        )
    items.sort(key=lambda item: str(item.get("name") or "").lower())
    return items


async def _resolve_workflow_name(request: Request, name: str):
    registry = request.app.state.registry
    requested = (name or "").strip()
    if not requested:
        raise _admin_error(400, "Missing workflow name")
    wf = await registry.get(requested)
    if wf:
        return wf
    for item in await registry.list():
        if item.name.lower() == requested.lower():
            return item
    load_error = await registry.get_load_error(requested)
    if load_error:
        raise _admin_error(400, f"Workflow '{load_error.name}' failed to load: {load_error.error}")
    raise _admin_error(404, "Workflow not found")


def _base_url(request: Request, cfg: Config) -> str:
    return (cfg.public_base_url or str(request.base_url)).rstrip("/")


def _abs_url(request: Request, cfg: Config, maybe_path: str) -> str:
    if not maybe_path:
        return ""
    if maybe_path.startswith("/"):
        return _base_url(request, cfg) + maybe_path
    return maybe_path


def _authorized_url(request: Request, cfg: Config, maybe_path: str) -> str:
    url = _abs_url(request, cfg, maybe_path)
    if not url or not cfg.api_token:
        return url
    secret = signing_secret(configured_secret=cfg.signed_url_secret, api_token=cfg.api_token)
    if not secret:
        return url
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    params = [(key, value) for key, value in params if key not in {"sig", "exp", "authorization", "api_key", "token", "access_token"}]
    params.extend(
        create_signed_query(
            path=parts.path,
            ttl_seconds=cfg.signed_url_ttl_seconds,
            secret=secret,
        ).items()
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def _rewrite_task_payload_urls(request: Request, cfg: Config, payload: dict[str, Any]) -> dict[str, Any]:
    copied = {"task": dict(payload.get("task") or {}), "outputs": []}
    job_id = str(copied["task"].get("job_id") or "")
    outputs: list[dict[str, Any]] = []
    for raw in payload.get("outputs") or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        filename = str(item.get("filename") or Path(str(item.get("url") or "")).name)
        if filename and job_id:
            item["url"] = _authorized_url(request, cfg, f"/runs/{job_id}/{filename}")
        outputs.append(item)
    copied["outputs"] = outputs
    raw_primary = str(copied["task"].get("url") or "")
    primary_name = Path(raw_primary).name if raw_primary else ""
    if primary_name and job_id:
        copied["task"]["url"] = _authorized_url(request, cfg, f"/runs/{job_id}/{primary_name}")
    return copied
