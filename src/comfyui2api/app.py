from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import re
import socket
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .admin_limiter import AdminAuthLimiter
from .admin_routes import create_admin_router
from .backend_store import BackendStore
from .config import Config, load_config
from .errors import InvalidParameterError
from .input_staging import stage_input_image
from .instance_pool import InstancePool
from .job_retention import run_job_retention_forever
from .job_scheduler import STAGED_PREFIX
from .job_store import JobStore
from .jobs import JobManager
from .model_catalog import (
    KindMismatchError,
    ModelCatalog,
    ModelNotFoundError,
    NoAvailableBackendError,
    WorkflowUnavailableError,
)
from .signed_urls import create_signed_query, has_valid_signature, signing_secret
from .util import (
    bearer_authorized,
    decode_data_url_base64,
    guess_media_type,
    utc_now_unix,
)
from .workflow_params import AUDIO_PARAMETER_ORDER, STANDARD_PARAMETER_ORDER
from .workflow_registry import WorkflowRegistry


logger = logging.getLogger(__name__)


def _openai_error(
    message: str,
    *,
    code: str = "invalid_request_error",
    error_code: str | None = None,
    http_status: int = 400,
    extra: Mapping[str, Any] | None = None,
) -> HTTPException:
    error = {"message": message, "type": code}
    if error_code:
        error["code"] = error_code
    if extra:
        error.update(extra)
    return HTTPException(
        status_code=http_status,
        detail={"error": error},
    )


def _require_auth(cfg: Config, authorization: str | None) -> None:
    if not cfg.api_token:
        return
    if not bearer_authorized(authorization or "", cfg.api_token):
        raise _openai_error("Unauthorized", code="invalid_api_key", http_status=401)


def _require_download_access(cfg: Config, request: Request, authorization: str | None) -> None:
    if not cfg.api_token:
        return
    secret = signing_secret(configured_secret=cfg.signed_url_secret, api_token=cfg.api_token)
    if has_valid_signature(path=request.url.path, query_params=request.query_params, secret=secret):
        return
    _require_auth(cfg, _auth_value_from_request(request, authorization))


def _uuid_now_hex() -> str:
    import uuid

    return uuid.uuid4().hex


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


def _auth_value_from_request(request: Request, authorization: str | None) -> str | None:
    header_value = (authorization or "").strip()
    if header_value:
        return header_value
    return _auth_value_from_query_params(request.query_params)


def _auth_value_from_ws(ws: WebSocket) -> str | None:
    header_value = (ws.headers.get("authorization") or "").strip()
    if header_value:
        return header_value
    return _auth_value_from_query_params(ws.query_params)


class _RequestBodyTooLargeError(RuntimeError):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self.size = size


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max(0, int(max_body_bytes))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.max_body_bytes <= 0:
            await self.app(scope, receive, send)
            return

        response_started = False
        seen = 0
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        content_length = headers.get("content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                declared = None
            if declared is not None and declared > self.max_body_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "message": f"Request body too large ({declared} bytes)",
                            "type": "invalid_request_error",
                        }
                    },
                )
                await response(scope, receive, send)
                return

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] != "http.request":
                return message
            body = message.get("body", b"")
            seen += len(body)
            if seen > self.max_body_bytes:
                raise _RequestBodyTooLargeError(seen)
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLargeError as exc:
            if response_started:
                return
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large ({exc.size} bytes)",
                        "type": "invalid_request_error",
                    }
                },
            )
            await response(scope, receive, send)


def _clean_optional_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return value


def _collect_standard_params(values: Mapping[str, Any], *, aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    alias_map = dict(aliases or {})
    params: dict[str, Any] = {}
    for key, raw_value in values.items():
        name = alias_map.get(key, key)
        if name not in STANDARD_PARAMETER_ORDER:
            continue
        value = _clean_optional_value(raw_value)
        if value is None:
            continue
        params[name] = value
    return params


async def _collect_workflow_request_params(
    *,
    spec: Any,
    values: Mapping[str, Any],
    store_input_image_bytes: Any,
    store_input_image_value: Any,
    skip: set[str] | None = None,
) -> dict[str, Any]:
    if spec is None or not getattr(spec, "parameters", None):
        return {}

    skipped = set(skip or ())
    params: dict[str, Any] = {}
    for name, definition in spec.parameters.items():
        if name in skipped or _is_edit_image_field_name(name) or name not in values:
            continue
        raw_value = values.get(name)
        if definition.type == "image":
            if hasattr(raw_value, "read"):
                data = await raw_value.read()
                if not data:
                    continue
                filename_hint = getattr(raw_value, "filename", None) or name
                params[name] = await store_input_image_bytes(data=data, filename_hint=filename_hint)
                continue
            cleaned = _clean_optional_value(raw_value)
            if cleaned is None:
                continue
            params[name] = await store_input_image_value(str(cleaned), filename_hint=name)
            continue

        cleaned = _clean_optional_value(raw_value)
        if cleaned is None:
            continue
        params[name] = cleaned
    return params


_EDIT_IMAGE_FIELD_NAMES = (
    "image",
    "image[]",
    "images",
    "input_reference",
    "input_reference[]",
    "input_references",
)
_EDIT_IMAGE_MAX = 16
_EDIT_INDEXED_IMAGE_RE = re.compile(r"^(?:image|input_reference)\[(\d+)\]$")
_EDIT_NUMBERED_IMAGE_RE = re.compile(r"^(?:image|input_reference)(\d+)$")
_EDIT_RESERVED_FIELDS = {
    "prompt",
    "model",
    "workflow",
    "negative_prompt",
    "response_format",
    "mask",
    "image",
    "image[]",
    "images",
    *STANDARD_PARAMETER_ORDER,
}


def _mapping_getlist(values: Any, name: str) -> list[Any]:
    if values is None:
        return []
    getlist = getattr(values, "getlist", None)
    if callable(getlist):
        raw_list = getlist(name)
        if raw_list is None:
            return []
        if isinstance(raw_list, list):
            return list(raw_list)
        return [raw_list]
    if not hasattr(values, "get"):
        return []
    raw = values.get(name)
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    return [raw]


def _is_empty_edit_image_item(item: Any) -> bool:
    if item is None:
        return True
    if isinstance(item, str):
        return not item.strip()
    filename = getattr(item, "filename", None)
    if filename is not None and str(filename).strip() == "" and not hasattr(item, "read"):
        return True
    return False


def _is_edit_image_field_name(name: str) -> bool:
    if name in _EDIT_IMAGE_FIELD_NAMES:
        return True
    if _EDIT_INDEXED_IMAGE_RE.match(name):
        return True
    match = _EDIT_NUMBERED_IMAGE_RE.match(name)
    return bool(match) and int(match.group(1)) >= 1


def _edit_image_item_marker(item: Any) -> Any:
    if isinstance(item, (str, bytes, int, float, bool)):
        return ("v", item)
    return ("id", id(item))


def _dedupe_edit_image_items(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique: list[Any] = []
    for item in items:
        marker = _edit_image_item_marker(item)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def _iter_mapping_pairs(values: Any) -> list[tuple[str, Any]]:
    if values is None:
        return []
    multi_items = getattr(values, "multi_items", None)
    if callable(multi_items):
        raw_pairs = list(multi_items())
        return [(str(key), value) for key, value in raw_pairs]
    items_fn = getattr(values, "items", None)
    if not callable(items_fn):
        return []
    pairs: list[tuple[str, Any]] = []
    for key, value in list(items_fn()):
        name = str(key)
        if isinstance(value, list):
            pairs.extend((name, item) for item in value)
        else:
            pairs.append((name, value))
    return pairs


def _collect_named_edit_image_items(values: Any) -> list[Any]:
    canonical: list[Any] = []
    indexed: list[tuple[int, Any]] = []
    numbered: list[tuple[int, Any]] = []
    pairs = _iter_mapping_pairs(values)
    if pairs:
        for name, value in pairs:
            if name in _EDIT_IMAGE_FIELD_NAMES:
                canonical.append(value)
                continue
            indexed_match = _EDIT_INDEXED_IMAGE_RE.match(name)
            if indexed_match:
                indexed.append((int(indexed_match.group(1)), value))
                continue
            numbered_match = _EDIT_NUMBERED_IMAGE_RE.match(name)
            if numbered_match:
                index = int(numbered_match.group(1))
                if index >= 1:
                    numbered.append((index, value))
    else:
        for name in _EDIT_IMAGE_FIELD_NAMES:
            canonical.extend(_mapping_getlist(values, name))
        if values is not None and hasattr(values, "keys"):
            for raw_name in list(values.keys()):
                name = str(raw_name)
                indexed_match = _EDIT_INDEXED_IMAGE_RE.match(name)
                if indexed_match:
                    indexed.extend((int(indexed_match.group(1)), item) for item in _mapping_getlist(values, name))
                    continue
                numbered_match = _EDIT_NUMBERED_IMAGE_RE.match(name)
                if numbered_match:
                    index = int(numbered_match.group(1))
                    if index >= 1:
                        numbered.extend((index, item) for item in _mapping_getlist(values, name))
    indexed.sort(key=lambda item: item[0])
    numbered.sort(key=lambda item: item[0])
    if canonical:
        numbered = [item for item in numbered if item[0] >= 2]
    return canonical + [value for _, value in indexed] + [value for _, value in numbered]


def collect_edit_image_items(*, values: Any, is_form: bool) -> list[Any]:
    if is_form:
        items = _collect_named_edit_image_items(values)
    else:
        items = []
        if values is not None and hasattr(values, "get"):
            for key in ("images", "input_references"):
                raw = values.get(key)
                if raw is None:
                    continue
                items.extend(list(raw) if isinstance(raw, list) else [raw])
        items.extend(_collect_named_edit_image_items(values))
    return [item for item in _dedupe_edit_image_items(items) if not _is_empty_edit_image_item(item)]


def extra_edit_image_params(image_rels: list[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {"inputImgCount": len(image_rels)}
    for index, rel in enumerate(image_rels[1:], start=2):
        extras[f"image{index}"] = rel
    return extras


def create_app() -> FastAPI:
    cfg = load_config()
    registry = WorkflowRegistry(cfg.workflows_dir)
    store = JobStore(cfg.database_path)
    backend = BackendStore(cfg.database_path)
    pool = InstancePool(cfg=cfg, store=backend)
    catalog = ModelCatalog(store=backend, pool=pool, registry=registry)
    jobs = JobManager(cfg=cfg, registry=registry, pool=pool, backend=backend, store=store)

    app = FastAPI(title="comfyui2api", version="0.1.0")
    app.add_middleware(MaxBodySizeMiddleware, max_body_bytes=cfg.max_body_bytes)
    app.state.cfg = cfg
    app.state.registry = registry
    app.state.jobs = jobs
    app.state.job_store = store
    app.state.backend = backend
    app.state.pool = pool
    app.state.catalog = catalog
    app.state.admin_limiter = AdminAuthLimiter()
    app.include_router(create_admin_router())

    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):  # type: ignore[override]
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return await http_exception_handler(request, exc)

    @app.exception_handler(InvalidParameterError)
    async def _invalid_parameter_handler(request: Request, exc: InvalidParameterError) -> JSONResponse:
        # create_job validates standard params for every endpoint; reuse the
        # OpenAI error envelope so the shape matches the endpoints' own errors.
        error = _openai_error(str(exc))
        return JSONResponse(status_code=error.status_code, content=error.detail)

    @app.on_event("startup")
    async def _startup() -> None:
        await store.init()
        await backend.init()
        await store.mark_unfinished_interrupted()
        await pool.start()
        await registry.load_all()
        if cfg.enable_workflow_watch:
            app.state.workflow_watch_task = asyncio.create_task(registry.watch_forever(), name="workflow-watch")
        if cfg.job_retention_seconds > 0 or cfg.max_jobs_in_memory > 0:
            app.state.job_retention_task = asyncio.create_task(
                run_job_retention_forever(
                    jobs,
                    interval_s=cfg.job_cleanup_interval_s,
                    ttl_seconds=cfg.job_retention_seconds,
                    max_jobs=cfg.max_jobs_in_memory,
                ),
                name="job-retention",
            )
        await jobs.start_workers()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        retention_task = getattr(app.state, "job_retention_task", None)
        if retention_task:
            retention_task.cancel()
            await asyncio.gather(retention_task, return_exceptions=True)
        t = getattr(app.state, "workflow_watch_task", None)
        if t:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        await jobs.stop_workers()
        await pool.aclose()
        await store.aclose()

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/runs/{job_id}/{output_name}")
    async def get_run_output(
        request: Request,
        job_id: str,
        output_name: str,
        authorization: Optional[str] = Header(default=None),
    ) -> Any:
        _require_download_access(cfg, request, authorization)
        job = await jobs.get_job(job_id)
        if not job:
            raise _openai_error("Job not found", http_status=404)
        if job.status != "completed":
            raise _openai_error("Output not ready", http_status=409)

        pick = None
        for item in job.outputs or []:
            if item.filename == output_name:
                pick = item
                break
        if pick is None:
            raise _openai_error("Output not found", http_status=404)

        path = (cfg.runs_dir / job_id / pick.filename).resolve()
        if not path.exists():
            raise _openai_error("Output file missing", http_status=500)

        return FileResponse(path=str(path), media_type=pick.media_type or guess_media_type(pick.filename), filename=pick.filename)

    async def _find_workflow_load_error(name: str):
        requested = (name or "").strip()
        if not requested:
            return None
        item = await registry.get_load_error(requested)
        if item:
            return item
        for load_error in await registry.list_load_errors():
            if load_error.name.lower() == requested.lower():
                return load_error
        return None

    async def _resolve_workflow_name(name: str):
        requested = (name or "").strip()
        if not requested:
            raise _openai_error("Missing workflow name", http_status=400)

        wf = await registry.get(requested)
        if wf:
            return wf
        for item in await registry.list():
            if item.name.lower() == requested.lower():
                return item

        if not requested.lower().endswith(".json"):
            maybe = requested + ".json"
            wf = await registry.get(maybe)
            if wf:
                return wf
            for item in await registry.list():
                if item.name.lower() == maybe.lower():
                    return item
            load_error = await _find_workflow_load_error(maybe)
            if load_error:
                raise _openai_error(
                    f"Workflow '{load_error.name}' failed to load: {load_error.error}",
                    http_status=400,
                )

        load_error = await _find_workflow_load_error(requested)
        if load_error:
            raise _openai_error(
                f"Workflow '{load_error.name}' failed to load: {load_error.error}",
                http_status=400,
            )

        raise _openai_error("Workflow not found", http_status=404)

    def _raise_catalog(exc: Exception) -> None:
        if isinstance(exc, ModelNotFoundError):
            raise _openai_error(
                str(exc),
                code="invalid_request_error",
                error_code="model_not_found",
                http_status=404,
            ) from exc
        if isinstance(exc, WorkflowUnavailableError):
            raise _openai_error(
                str(exc),
                code="invalid_request_error",
                error_code="workflow_unavailable",
                http_status=400,
            ) from exc
        if isinstance(exc, NoAvailableBackendError):
            raise _openai_error(
                str(exc),
                code="server_error",
                error_code="no_available_backend",
                http_status=503,
            ) from exc
        if isinstance(exc, KindMismatchError):
            raise _openai_error(str(exc), http_status=400) from exc
        raise exc

    async def _resolve_public_model(*, model: str, kind: str | None, has_image: bool):
        try:
            return await catalog.resolve(slug=model, kind=kind, has_image=has_image)
        except (ModelNotFoundError, WorkflowUnavailableError, NoAvailableBackendError, KindMismatchError) as exc:
            _raise_catalog(exc)
            raise

    async def _store_input_image_bytes(*, data: bytes, filename_hint: str | None) -> str:
        try:
            path = stage_input_image(
                runs_dir=cfg.runs_dir,
                data=data,
                filename_hint=filename_hint,
                max_bytes=cfg.max_image_bytes,
                name_prefix=_uuid_now_hex(),
            )
        except ValueError as exc:
            raise _openai_error(str(exc), http_status=413) from exc
        return f"{STAGED_PREFIX}{path}"

    def _is_global_public_ip(host: str) -> bool:
        try:
            info = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except Exception:
            return False
        if not info:
            return False
        for item in info:
            ip_str = item[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return False
            if not ip.is_global:
                return False
        return True

    async def _download_image_url(url: str) -> bytes:
        u = httpx.URL(url)
        if u.scheme not in {"http", "https"}:
            raise _openai_error("Only http/https image URLs are supported", http_status=400)
        if not u.host:
            raise _openai_error("Invalid image URL", http_status=400)
        if not _is_global_public_ip(u.host):
            raise _openai_error("Blocked image URL host", http_status=400)

        timeout = httpx.Timeout(timeout=cfg.http_timeout_s)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            current = u
            for _ in range(3):
                async with client.stream("GET", current, headers={"Accept": "image/*,*/*"}) as resp:
                    if 300 <= resp.status_code < 400 and resp.headers.get("location"):
                        nxt = httpx.URL(resp.headers["location"])
                        current = nxt if nxt.scheme else current.join(nxt)
                        if current.scheme not in {"http", "https"} or not current.host or not _is_global_public_ip(current.host):
                            raise _openai_error("Blocked image URL redirect", http_status=400)
                        continue

                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        raise _openai_error(f"Failed to download image: HTTP {e.response.status_code}", http_status=400) from e

                    content_type = (resp.headers.get("content-type") or "").lower()
                    if content_type and not content_type.startswith("image/"):
                        raise _openai_error("URL did not return an image", http_status=400)

                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > max(1, int(cfg.max_image_bytes)):
                            raise _openai_error("Image too large", http_status=413)
                    return bytes(buf)

            raise _openai_error("Too many redirects downloading image", http_status=400)

    async def _store_input_image_value(image_value: str, *, filename_hint: str | None = None) -> str:
        s = (image_value or "").strip()
        if not s:
            raise _openai_error("Missing 'image'", http_status=400)
        if s.startswith("http://") or s.startswith("https://"):
            data = await _download_image_url(s)
            return await _store_input_image_bytes(data=data, filename_hint=filename_hint or "image")
        data = decode_data_url_base64(s)
        return await _store_input_image_bytes(data=data, filename_hint=filename_hint)

    def _base_url(request: Request) -> str:
        return (cfg.public_base_url or str(request.base_url)).rstrip("/")

    def _abs_url(request: Request, maybe_path: str) -> str:
        if not maybe_path:
            return ""
        if maybe_path.startswith("/"):
            return _base_url(request) + maybe_path
        return maybe_path

    def _authorized_url_parts(request: Request, maybe_path: str, authorization: str | None) -> tuple[str, int | None]:
        url = _abs_url(request, maybe_path)
        if not url or not cfg.api_token:
            return url, None

        secret = signing_secret(configured_secret=cfg.signed_url_secret, api_token=cfg.api_token)
        if not secret:
            return url, None

        parts = urlsplit(url)
        params = parse_qsl(parts.query, keep_blank_values=True)
        params = [(key, value) for key, value in params if key not in {"sig", "exp", "authorization", "api_key", "token", "access_token"}]
        signed_query = create_signed_query(
            path=parts.path,
            ttl_seconds=cfg.signed_url_ttl_seconds,
            secret=secret,
        )
        params.extend(signed_query.items())
        signed_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))
        return signed_url, int(signed_query["exp"])

    def _authorized_url(request: Request, maybe_path: str, authorization: str | None) -> str:
        url, _expires_at = _authorized_url_parts(request, maybe_path, authorization)
        return url

    def _output_relpath(job_id: str, filename: str) -> str:
        return f"/runs/{job_id}/{filename}"

    def _authorized_output_url(request: Request, *, job_id: str, output: Mapping[str, Any], authorization: str | None) -> str:
        filename = str(output.get("filename") or "").strip()
        if not filename:
            raw_url = str(output.get("url") or "").strip()
            filename = Path(raw_url).name if raw_url else ""
        if not filename:
            return ""
        return _authorized_url(request, _output_relpath(job_id, filename), authorization)

    def _rewrite_public_job_urls(request: Request, public: Dict[str, Any], authorization: str | None) -> Dict[str, Any]:
        rewritten = dict(public)
        outputs = []
        raw_outputs = public.get("outputs") or []
        for item in raw_outputs if isinstance(raw_outputs, list) else []:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            copied["url"] = _authorized_output_url(
                request,
                job_id=str(public.get("job_id") or ""),
                output=copied,
                authorization=authorization,
            )
            outputs.append(copied)
        rewritten["outputs"] = outputs

        raw_primary = str(public.get("url") or "").strip()
        primary_name = Path(raw_primary).name if raw_primary else ""
        if primary_name:
            rewritten["url"] = _authorized_url(request, _output_relpath(str(public.get("job_id") or ""), primary_name), authorization)
        elif outputs:
            rewritten["url"] = outputs[0].get("url")
        else:
            rewritten["url"] = None
        return rewritten

    def _response_output_urls(
        request: Request,
        *,
        job_id: str,
        outputs: list[dict[str, Any]],
        authorization: str | None,
    ) -> list[str]:
        urls: list[str] = []
        for item in outputs:
            url = _authorized_output_url(request, job_id=job_id, output=item, authorization=authorization)
            if url:
                urls.append(url)
        return urls

    def _output_filename(item: Mapping[str, Any]) -> str:
        filename = str(item.get("filename") or "").strip()
        if filename:
            return filename
        raw_url = str(item.get("url") or "").strip()
        return Path(raw_url).name if raw_url else ""

    def _pick_audio_output(outputs: list[dict[str, Any]]) -> Mapping[str, Any] | None:
        # Prefer an actual audio output (MIME or extension) over "whatever
        # came first": a TTS workflow may also emit ancillary files (e.g. a
        # preview image), and blind outputs[0] could return those instead.
        def _is_audio(item: Mapping[str, Any]) -> bool:
            media = str(item.get("media_type") or "").lower()
            if media.startswith("audio/"):
                return True
            return _output_filename(item).lower().endswith(
                (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
            )

        for item in outputs:
            if _is_audio(item):
                return item
        return outputs[0] if outputs else None

    def _b64_json_data(job_id: str, outputs: list[dict[str, Any]]) -> list[dict[str, str]]:
        encoded: list[dict[str, str]] = []
        for item in outputs:
            filename = _output_filename(item)
            if not filename:
                continue
            path = Path(cfg.runs_dir) / job_id / filename
            if not path.is_file():
                continue
            encoded.append({"b64_json": base64.b64encode(path.read_bytes()).decode("ascii")})
        if not encoded:
            raise _openai_error("No outputs produced", http_status=500)
        return encoded

    def _normalize_image_response_format(value: Any) -> str:
        raw = str(value or "url").strip().lower()
        if not raw:
            return "url"
        if raw in {"b64_json", "b64", "base64", "base64_json"}:
            return "b64_json"
        return "url"

    def _extract_chat_message_content(content: Any) -> tuple[list[str], str | None]:
        texts: list[str] = []
        image_value: str | None = None
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned:
                texts.append(cleaned)
            return texts, None
        if not isinstance(content, list):
            return texts, None

        for part in content:
            if isinstance(part, str):
                cleaned = part.strip()
                if cleaned:
                    texts.append(cleaned)
                continue
            if not isinstance(part, dict):
                continue

            part_type = str(part.get("type") or "").strip().lower()
            if part_type in {"text", "input_text"}:
                raw_text = part.get("text")
                if raw_text is None and part_type == "input_text":
                    raw_text = part.get("input_text")
                cleaned = str(raw_text or "").strip()
                if cleaned:
                    texts.append(cleaned)
                continue

            if image_value is not None:
                continue
            if part_type not in {"image_url", "input_image", "image"}:
                continue

            raw_image: Any = part.get("image_url")
            if raw_image is None:
                raw_image = part.get("input_image")
            if raw_image is None:
                raw_image = part.get("url")
            if raw_image is None:
                raw_image = part.get("image")
            if isinstance(raw_image, dict):
                raw_image = raw_image.get("url") or raw_image.get("image_url")
            cleaned_image = str(raw_image or "").strip()
            if cleaned_image:
                image_value = cleaned_image

        return texts, image_value

    def _extract_chat_prompt_and_image(messages: Any) -> tuple[str, str | None]:
        prompt = ""
        image_value: str | None = None
        if not isinstance(messages, list):
            return prompt, image_value

        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role != "user":
                continue
            texts, maybe_image = _extract_chat_message_content(item.get("content"))
            if not prompt and texts:
                prompt = "\n\n".join(texts).strip()
            if image_value is None and maybe_image:
                image_value = maybe_image
            if prompt and image_value is not None:
                break

        return prompt, image_value

    def _chat_completion_response(*, model: str, content_payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl_{_uuid_now_hex()}",
            "object": "chat.completion",
            "created": utc_now_unix(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(dict(content_payload), ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.get("/v1/models")
    async def openai_models(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        _require_auth(cfg, authorization)
        return {"object": "list", "data": [item.as_openai() for item in await catalog.list_public()]}

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        request: Request,
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _require_auth(cfg, authorization)
        if body.get("stream") is True:
            raise _openai_error("Streaming is not supported for /v1/chat/completions", http_status=400)

        requested_model = str(body.get("model") or "").strip()
        if not requested_model:
            raise _openai_error("Missing 'model'", error_code="model_not_found", http_status=404)

        prompt = str(body.get("prompt") or "").strip()
        image_value = _clean_optional_value(body.get("image"))
        if image_value is not None:
            image_value = str(image_value).strip()
        if not prompt or not image_value:
            message_prompt, message_image = _extract_chat_prompt_and_image(body.get("messages"))
            if not prompt:
                prompt = message_prompt
            if not image_value and message_image:
                image_value = message_image
        if not prompt:
            raise _openai_error("Missing prompt text in 'messages'")

        resolved = await _resolve_public_model(
            model=requested_model,
            kind=str(body.get("kind") or "").strip() or None,
            has_image=bool(image_value),
        )
        wf = resolved.workflow
        kind = resolved.kind
        negative_prompt = str(body.get("negative_prompt") or "").strip()
        standard_params = _collect_standard_params({key: body.get(key) for key in STANDARD_PARAMETER_ORDER if key in body})
        seconds_value = _clean_optional_value(body.get("seconds"))
        if seconds_value is not None and "duration" not in standard_params:
            standard_params["duration"] = seconds_value

        image_rel = ""
        if image_value:
            image_rel = await _store_input_image_value(str(image_value), filename_hint="chat_input")

        job = await jobs.create_job(
            kind=kind,
            workflow=wf.name,
            platform="Chat",
            requested_model=requested_model,
            seconds=str(seconds_value or ""),
            size=str(body.get("size") or "").strip(),
            quality=str(body.get("quality") or "").strip() or "standard",
            metadata="",
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_rel,
            standard_params=standard_params,
        )

        public_model = requested_model
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            payload: dict[str, Any] = {
                "type": "generation_job",
                "kind": kind,
                "job_id": job.job_id,
                "status": "pending",
            }
            if kind.endswith("video"):
                payload["video_id"] = _video_id(job.job_id)
            return _chat_completion_response(model=public_model, content_payload=payload)

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        response_format = _normalize_image_response_format(body.get("response_format"))

        payload = {
            "type": "generation_result",
            "kind": kind,
            "job_id": str(done.get("job_id") or job.job_id),
            "status": "completed",
        }
        if kind.endswith("video"):
            payload["video_id"] = _video_id(str(done.get("job_id") or job.job_id))

        if response_format == "b64_json" and kind in {"txt2img", "img2img"}:
            payload["data"] = _b64_json_data(str(done.get("job_id") or job.job_id), outputs)
        else:
            urls = _response_output_urls(
                request,
                job_id=str(done.get("job_id") or job.job_id),
                outputs=outputs,
                authorization=authorization,
            )
            payload["data"] = [{"url": u} for u in urls]

        return _chat_completion_response(model=public_model, content_payload=payload)

    @app.post("/v1/jobs")
    async def submit_job(
        request: Request,
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _require_auth(cfg, authorization)

        requested_model = str(body.get("model") or "").strip()
        if not requested_model:
            raise _openai_error("Missing 'model'", error_code="model_not_found", http_status=404)
        if str(body.get("workflow") or "").strip():
            raise _openai_error("Use 'model' (external model id); 'workflow' is not a routing key", http_status=400)
        kind = str(body.get("kind") or "").strip() or None
        image_items = collect_edit_image_items(values=body, is_form=False)
        has_image = bool(image_items or body.get("image_base64"))
        resolved = await _resolve_public_model(model=requested_model, kind=kind, has_image=has_image)
        wf = resolved.workflow
        workflow = wf.name
        kind = resolved.kind
        prompt = str(body.get("prompt") or "").strip()
        negative_prompt = str(body.get("negative_prompt") or "").strip()

        prompt_node = str(body.get("prompt_node") or "").strip()
        negative_prompt_node = str(body.get("negative_prompt_node") or "").strip()
        image_node = str(body.get("image_node") or "").strip()

        image_rels: list[str] = []
        for index, item in enumerate(image_items, start=1):
            stored = await _store_edit_image_item(item, filename_hint=f"image{index}")
            if stored:
                image_rels.append(stored)
        if not image_rels and body.get("image_base64"):
            img_bytes = decode_data_url_base64(str(body.get("image_base64") or ""))
            filename_hint = str(body.get("image_filename") or "") or None
            image_rels.append(await _store_input_image_bytes(data=img_bytes, filename_hint=filename_hint))
        image_rel = image_rels[0] if image_rels else ""

        overrides: list[tuple[str, str, Any]] = []
        raw_overrides = body.get("overrides")
        if isinstance(raw_overrides, dict):
            for k, v in raw_overrides.items():
                if not isinstance(k, str) or "." not in k:
                    continue
                node_id, input_key = k.split(".", 1)
                overrides.append((node_id.strip(), input_key.strip(), v))

        standard_params = _collect_standard_params(
            {key: body.get(key) for key in STANDARD_PARAMETER_ORDER if key in body},
            aliases={"seconds": "duration"},
        )
        seconds_value = _clean_optional_value(body.get("seconds"))
        if seconds_value is not None and "duration" not in standard_params:
            standard_params["duration"] = seconds_value
        standard_params.update(
            await _collect_workflow_request_params(
                spec=wf.parameter_spec,
                values=body,
                store_input_image_bytes=_store_input_image_bytes,
                store_input_image_value=_store_input_image_value,
                skip={
                    *STANDARD_PARAMETER_ORDER,
                    "kind",
                    "workflow",
                    "model",
                    "prompt",
                    "negative_prompt",
                    "image",
                    "image_base64",
                    "image_filename",
                    "input_reference",
                    "input_reference2",
                    "prompt_node",
                    "negative_prompt_node",
                    "image_node",
                    "overrides",
                    "seconds",
                    "inputImgCount",
                },
            )
        )
        spec_parameters = getattr(getattr(wf, "parameter_spec", None), "parameters", None) or {}
        if image_rels or "inputImgCount" in spec_parameters:
            standard_params.update(extra_edit_image_params(image_rels))

        job = await jobs.create_job(
            kind=kind,
            workflow=workflow,
            platform="Native",
            requested_model=requested_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_rel,
            prompt_node=prompt_node,
            negative_prompt_node=negative_prompt_node,
            image_node=image_node,
            overrides=overrides,
            standard_params=standard_params,
        )

        base = _base_url(request)
        job_url = f"{base}/v1/jobs/{job.job_id}"
        ws_url = f"{base.replace('http://', 'ws://').replace('https://', 'wss://')}/v1/jobs/{job.job_id}/ws"
        return {"job": jobs.public_job(job), "job_url": job_url, "ws_url": ws_url}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(request: Request, job_id: str, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        _require_auth(cfg, authorization)
        job = await jobs.get_job(job_id)
        if not job:
            raise _openai_error("Job not found", http_status=404)
        public = _rewrite_public_job_urls(request, jobs.public_job(job), authorization)
        return {"job": public}

    @app.get("/v1/queue")
    async def get_queue(request: Request, authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        _require_auth(cfg, authorization)
        items = await jobs.list_jobs(limit=200)
        counts: Dict[str, int] = {}
        for j in items:
            counts[j.status] = counts.get(j.status, 0) + 1
        result_items = []
        for j in items:
            result_items.append(_rewrite_public_job_urls(request, jobs.public_job(j), authorization))
        return {"counts": counts, "items": result_items}

    @app.websocket("/v1/jobs/{job_id}/ws")
    async def job_ws(ws: WebSocket, job_id: str) -> None:
        try:
            _require_auth(cfg, _auth_value_from_ws(ws))
        except HTTPException as exc:
            # Accept first, then send a structured JSON error frame and close.
            # Closing an unaccepted socket makes uvicorn answer the handshake
            # with HTTP 403, so browsers only see a generic transport error.
            await ws.accept()
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "data": {"status": exc.status_code, "message": "Unauthorized"},
                    }
                )
            except Exception:
                pass
            await ws.close(code=1008, reason="Unauthorized")
            return
        await ws.accept()
        job = await jobs.get_job(job_id)
        if not job:
            await ws.send_json({"type": "error", "data": {"message": "Job not found"}})
            await ws.close(code=1008)
            return

        await jobs.subscribe(job_id, ws)
        try:
            await ws.send_json({"type": "job_snapshot", "data": jobs.public_job(job)})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await jobs.unsubscribe(job_id, ws)

    async def _openai_wait(job_id: str) -> Dict[str, Any]:
        job = await jobs.get_job(job_id)
        if not job:
            raise _openai_error("Job not found", http_status=404)
        await job.done.wait()
        job = await jobs.get_job(job_id)
        if not job:
            raise _openai_error("Job not found", http_status=404)
        if job.status != "completed":
            error_message = job.error or "Job failed"
            extra = {"job_id": job_id}
            if "ComfyApiError: ComfyUI " in error_message:
                extra["upstream"] = "comfyui"
                extra["instance_slug"] = job.instance_slug or None
            raise _openai_error(
                error_message,
                code="server_error",
                http_status=500,
                extra=extra,
            )
        return jobs.public_job(job)

    @app.post("/v1/images/generations")
    async def openai_images_generations(
        request: Request,
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Any:
        _require_auth(cfg, authorization)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise _openai_error("Missing 'prompt'")
        resolved = await _resolve_public_model(
            model=str(body.get("model") or "").strip(),
            kind="txt2img",
            has_image=False,
        )
        wf = resolved.workflow
        workflow = wf.name
        negative_prompt = str(body.get("negative_prompt") or "").strip()
        response_format = _normalize_image_response_format(body.get("response_format"))
        standard_params = _collect_standard_params(
            {key: body.get(key) for key in (*STANDARD_PARAMETER_ORDER, "n") if key in body}
        )

        job = await jobs.create_job(
            kind="txt2img",
            workflow=workflow,
            platform="OpenAI",
            requested_model=resolved.record.slug,
            prompt=prompt,
            negative_prompt=negative_prompt,
            standard_params=standard_params,
        )
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            return {"job_id": job.job_id, "status": "pending"}

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        urls = _response_output_urls(
            request,
            job_id=str(done.get("job_id") or job.job_id),
            outputs=outputs,
            authorization=authorization,
        )
        if response_format == "b64_json":
            return {
                "created": utc_now_unix(),
                "data": _b64_json_data(str(done.get("job_id") or job.job_id), outputs),
            }
        return {"created": utc_now_unix(), "data": [{"url": u} for u in urls]}

    async def _parse_images_edits_payload(request: Request) -> Dict[str, Any]:
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if content_type == "application/json":
            try:
                body = await request.json()
            except Exception as e:
                raise _openai_error("Invalid JSON body", http_status=400) from e
            if not isinstance(body, dict):
                raise _openai_error("JSON body must be an object", http_status=400)
            return {
                "prompt": str(body.get("prompt") or "").strip(),
                "model": str(body.get("model") or body.get("workflow") or "").strip(),
                "negative_prompt": str(body.get("negative_prompt") or "").strip(),
                "response_format": body.get("response_format"),
                "image_items": collect_edit_image_items(values=body, is_form=False),
                "raw_values": body,
            }

        form = await request.form()
        return {
            "prompt": str(form.get("prompt") or "").strip(),
            "model": str(form.get("model") or form.get("workflow") or "").strip(),
            "negative_prompt": str(form.get("negative_prompt") or "").strip(),
            "response_format": form.get("response_format"),
            "image_items": collect_edit_image_items(values=form, is_form=True),
            "raw_values": form,
        }

    async def _store_edit_image_item(item: Any, *, filename_hint: str) -> str | None:
        if hasattr(item, "read"):
            data = await item.read()
            if not data:
                return None
            hint = getattr(item, "filename", None) or filename_hint
            return await _store_input_image_bytes(data=data, filename_hint=hint)
        cleaned = _clean_optional_value(item)
        if cleaned is None:
            return None
        return await _store_input_image_value(str(cleaned), filename_hint=filename_hint)

    @app.post("/v1/images/edits")
    async def openai_images_edits(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Any:
        _require_auth(cfg, authorization)
        payload = await _parse_images_edits_payload(request)
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise _openai_error("Missing 'prompt'")
        image_items = list(payload.get("image_items") or [])
        if len(image_items) > _EDIT_IMAGE_MAX:
            raise _openai_error(f"Too many images: max {_EDIT_IMAGE_MAX}", http_status=400)

        image_rels: list[str] = []
        for index, item in enumerate(image_items, start=1):
            stored = await _store_edit_image_item(item, filename_hint=f"image{index}")
            if stored:
                image_rels.append(stored)
        if not image_rels:
            raise _openai_error("Missing 'image'")

        resolved = await _resolve_public_model(
            model=str(payload.get("model") or "").strip(),
            kind="img2img",
            has_image=True,
        )
        wf = resolved.workflow
        raw_values = payload.get("raw_values") or {}
        standard_params = _collect_standard_params(
            {key: raw_values.get(key) for key in (*STANDARD_PARAMETER_ORDER, "n") if hasattr(raw_values, "get")}
        )
        standard_params.update(
            await _collect_workflow_request_params(
                spec=wf.parameter_spec,
                values=raw_values,
                store_input_image_bytes=_store_input_image_bytes,
                store_input_image_value=_store_input_image_value,
                skip=_EDIT_RESERVED_FIELDS,
            )
        )
        standard_params.update(extra_edit_image_params(image_rels))
        job = await jobs.create_job(
            kind="img2img",
            workflow=wf.name,
            platform="OpenAI",
            requested_model=resolved.record.slug,
            prompt=prompt,
            negative_prompt=str(payload.get("negative_prompt") or "").strip(),
            image=image_rels[0],
            standard_params=standard_params,
        )
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            return {"job_id": job.job_id, "status": "pending"}

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        urls = _response_output_urls(
            request,
            job_id=str(done.get("job_id") or job.job_id),
            outputs=outputs,
            authorization=authorization,
        )
        if _normalize_image_response_format(payload.get("response_format")) == "b64_json":
            return {
                "created": utc_now_unix(),
                "data": _b64_json_data(str(done.get("job_id") or job.job_id), outputs),
            }
        return {"created": utc_now_unix(), "data": [{"url": u} for u in urls]}

    @app.post("/v1/images/variations")
    async def openai_images_variations(
        request: Request,
        image: UploadFile = File(...),
        model: str = Form(""),
        workflow: str = Form(""),
        response_format: str = Form("url"),
        size: str = Form(""),
        width: str = Form(""),
        height: str = Form(""),
        steps: str = Form(""),
        cfg_scale: str = Form("", alias="cfg"),
        seed: str = Form(""),
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Any:
        _require_auth(cfg, authorization)
        resolved = await _resolve_public_model(model=(model or "").strip(), kind="img2img", has_image=True)
        wf = resolved.workflow
        raw = await image.read()
        image_rel = await _store_input_image_bytes(data=raw, filename_hint=image.filename)

        standard_params = _collect_standard_params(
            {
                "size": size,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg_scale,
                "seed": seed,
            }
        )
        job = await jobs.create_job(
            kind="img2img",
            workflow=wf.name,
            platform="OpenAI",
            requested_model=resolved.record.slug,
            prompt="",
            image=image_rel,
            standard_params=standard_params,
        )
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            return {"job_id": job.job_id, "status": "pending"}

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        urls = _response_output_urls(
            request,
            job_id=str(done.get("job_id") or job.job_id),
            outputs=outputs,
            authorization=authorization,
        )
        if _normalize_image_response_format(response_format) == "b64_json":
            return {
                "created": utc_now_unix(),
                "data": _b64_json_data(str(done.get("job_id") or job.job_id), outputs),
            }
        return {"created": utc_now_unix(), "data": [{"url": u} for u in urls]}

    @app.post("/v1/videos/generations")
    async def openai_videos_generations(
        request: Request,
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Any:
        _require_auth(cfg, authorization)
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise _openai_error("Missing 'prompt'")
        resolved = await _resolve_public_model(
            model=str(body.get("model") or "").strip(),
            kind="txt2video",
            has_image=False,
        )
        wf = resolved.workflow
        workflow = wf.name

        negative_prompt = str(body.get("negative_prompt") or "").strip()
        standard_params = _collect_standard_params({key: body.get(key) for key in STANDARD_PARAMETER_ORDER if key in body})
        seconds_value = _clean_optional_value(body.get("seconds"))
        if seconds_value is not None and "duration" not in standard_params:
            standard_params["duration"] = seconds_value
        job = await jobs.create_job(
            kind="txt2video",
            workflow=workflow,
            platform="OpenAI",
            requested_model=resolved.record.slug,
            prompt=prompt,
            negative_prompt=negative_prompt,
            standard_params=standard_params,
        )
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            return {"job_id": job.job_id, "status": "pending"}

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        urls = _response_output_urls(
            request,
            job_id=str(done.get("job_id") or job.job_id),
            outputs=outputs,
            authorization=authorization,
        )
        return {"created": utc_now_unix(), "data": [{"url": u} for u in urls]}

    @app.post("/v1/videos/edits")
    async def openai_videos_edits(
        request: Request,
        image: UploadFile = File(...),
        prompt: str = Form(""),
        model: str = Form(""),
        workflow: str = Form(""),
        negative_prompt: str = Form(""),
        size: str = Form(""),
        fps: str = Form(""),
        duration: str = Form(""),
        frames: str = Form(""),
        width: str = Form(""),
        height: str = Form(""),
        steps: str = Form(""),
        cfg_scale: str = Form("", alias="cfg"),
        seed: str = Form(""),
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Any:
        _require_auth(cfg, authorization)
        form = await request.form()
        image_items = collect_edit_image_items(values=form, is_form=True)
        if not image_items and image is not None:
            image_items = [image]
        resolved = await _resolve_public_model(model=(model or "").strip(), kind="img2video", has_image=True)
        wf = resolved.workflow
        image_rels: list[str] = []
        for index, item in enumerate(image_items, start=1):
            stored = await _store_edit_image_item(item, filename_hint=f"image{index}")
            if stored:
                image_rels.append(stored)
        if not image_rels:
            raise _openai_error("Missing 'image'")
        image_rel = image_rels[0]

        standard_params = _collect_standard_params(
            {
                "size": size,
                "fps": fps,
                "duration": duration,
                "frames": frames,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg_scale,
                "seed": seed,
            }
        )
        standard_params.update(extra_edit_image_params(image_rels))
        job = await jobs.create_job(
            kind="img2video",
            workflow=wf.name,
            platform="OpenAI",
            requested_model=resolved.record.slug,
            prompt=(prompt or "").strip(),
            negative_prompt=(negative_prompt or "").strip(),
            image=image_rel,
            standard_params=standard_params,
        )
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            return {"job_id": job.job_id, "status": "pending"}

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        urls = _response_output_urls(
            request,
            job_id=str(done.get("job_id") or job.job_id),
            outputs=outputs,
            authorization=authorization,
        )
        return {"created": utc_now_unix(), "data": [{"url": u} for u in urls]}

    def _as_job_id_from_video_id(video_id: str) -> str:
        raw = (video_id or "").strip()
        if raw.startswith("video_"):
            return raw[len("video_") :]
        return raw

    def _video_id(job_id: str) -> str:
        return f"video_{job_id}"

    def _progress_percent(job_progress: Dict[str, Any], *, completed: bool) -> int:
        if completed:
            return 100
        if not isinstance(job_progress, dict):
            return 0
        try:
            value = float(job_progress.get("value") or 0)
            total = float(job_progress.get("max") or 0)
        except Exception:
            return 0
        if total <= 0:
            return 0
        pct = int((value / total) * 100.0)
        return max(0, min(99, pct))

    async def _workflow_from_model_or_default(*, kind: str, model: str, has_image: bool):
        resolved = await _resolve_public_model(model=model, kind=kind, has_image=has_image)
        return resolved

    async def _parse_videos_create_payload(request: Request) -> Dict[str, Any]:
        content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if content_type == "application/json":
            try:
                body = await request.json()
            except Exception as e:
                raise _openai_error("Invalid JSON body", http_status=400) from e
            if not isinstance(body, dict):
                raise _openai_error("JSON body must be an object", http_status=400)

            raw_metadata = body.get("metadata")
            if isinstance(raw_metadata, str):
                metadata = raw_metadata.strip()
            elif raw_metadata is None:
                metadata = ""
            else:
                metadata = json.dumps(raw_metadata, ensure_ascii=False)

            return {
                "prompt": str(body.get("prompt") or "").strip(),
                "model": str(body.get("model") or body.get("workflow") or "").strip(),
                "seconds": str(
                    _clean_optional_value(body.get("seconds"))
                    or _clean_optional_value(body.get("duration"))
                    or ""
                ).strip(),
                "size": str(body.get("size") or "").strip(),
                "fps": str(body.get("fps") or "").strip(),
                "frames": str(body.get("frames") or "").strip(),
                "width": str(body.get("width") or "").strip(),
                "height": str(body.get("height") or "").strip(),
                "quality": str(body.get("quality") or "").strip(),
                "metadata": metadata,
                "image_items": collect_edit_image_items(values=body, is_form=False),
                "raw_values": body,
            }

        form = await request.form()
        return {
            "prompt": str(form.get("prompt") or "").strip(),
            "model": str(form.get("model") or form.get("workflow") or "").strip(),
            "seconds": str(form.get("seconds") or form.get("duration") or "").strip(),
            "size": str(form.get("size") or "").strip(),
            "fps": str(form.get("fps") or "").strip(),
            "frames": str(form.get("frames") or "").strip(),
            "width": str(form.get("width") or "").strip(),
            "height": str(form.get("height") or "").strip(),
            "quality": str(form.get("quality") or "").strip(),
            "metadata": str(form.get("metadata") or "").strip(),
            "image_items": collect_edit_image_items(values=form, is_form=True),
            "raw_values": form,
        }

    @app.post("/v1/videos", status_code=200)
    async def openai_videos_create(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _require_auth(cfg, authorization)

        payload = await _parse_videos_create_payload(request)
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise _openai_error("Missing 'prompt'")

        image_items = list(payload.get("image_items") or [])
        image_rels: list[str] = []
        for index, item in enumerate(image_items, start=1):
            stored = await _store_edit_image_item(item, filename_hint=f"image{index}")
            if stored:
                image_rels.append(stored)
        image_rel = image_rels[0] if image_rels else ""
        kind = "img2video" if image_rel else "txt2video"

        resolved = await _workflow_from_model_or_default(
            kind=kind,
            model=str(payload.get("model") or ""),
            has_image=bool(image_rel),
        )
        wf = resolved.workflow
        workflow = wf.name
        requested_model = resolved.record.slug
        raw_values = payload.get("raw_values") or {}
        standard_params = _collect_standard_params(
            {
                "duration": payload.get("seconds"),
                "size": payload.get("size"),
                "fps": payload.get("fps"),
                "frames": payload.get("frames"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "seed": raw_values.get("seed") if hasattr(raw_values, "get") else None,
            }
        )
        standard_params.update(
            await _collect_workflow_request_params(
                spec=wf.parameter_spec,
                values=payload.get("raw_values") or {},
                store_input_image_bytes=_store_input_image_bytes,
                store_input_image_value=_store_input_image_value,
                skip={
                    *STANDARD_PARAMETER_ORDER,
                    "prompt",
                    "model",
                    "workflow",
                    "seconds",
                    "duration",
                    "size",
                    "fps",
                    "frames",
                    "width",
                    "height",
                    "quality",
                    "metadata",
                    "input_reference",
                    "input_reference2",
                    "image",
                    "inputImgCount",
                },
            )
        )
        spec_parameters = getattr(getattr(wf, "parameter_spec", None), "parameters", None) or {}
        if image_rels or "inputImgCount" in spec_parameters:
            standard_params.update(extra_edit_image_params(image_rels))
        job = await jobs.create_job(
            kind=kind,
            workflow=workflow,
            platform="OpenAI",
            requested_model=requested_model,
            seconds=str(payload.get("seconds") or "").strip(),
            size=str(payload.get("size") or "").strip(),
            quality=str(payload.get("quality") or "").strip() or "standard",
            metadata=str(payload.get("metadata") or "").strip(),
            prompt=prompt,
            image=image_rel,
            standard_params=standard_params,
        )

        return {
            "id": _video_id(job.job_id),
            "object": "video",
            "model": job.requested_model or workflow,
            "created_at": job.created_at,
            "status": "queued",
            "progress": 0,
        }

    @app.get("/v1/videos/{video_id}")
    async def openai_videos_get(
        request: Request,
        video_id: str,
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _require_auth(cfg, authorization)
        job_id = _as_job_id_from_video_id(video_id)
        job = await jobs.get_job(job_id)
        if not job:
            raise _openai_error("Video not found", http_status=404)

        status = "queued"
        if job.status == "running":
            status = "in_progress"
        elif job.status == "completed":
            status = "completed"
        elif job.status == "failed":
            status = "failed"

        completed = status == "completed"
        progress = _progress_percent(job.progress or {}, completed=completed)
        if status == "failed":
            progress = 0

        seconds = (job.seconds or "").strip() or "4"
        size = (job.size or "").strip() or "720x1280"
        quality = (job.quality or "").strip() or "standard"

        url = None
        expires_at = None
        if completed:
            url, expires_at = _authorized_url_parts(request, f"/v1/videos/{video_id}/content", authorization)

        err = None
        if status == "failed":
            err = {"message": job.error or "failed", "type": "server_error"}

        return {
            "id": _video_id(job.job_id),
            "object": "video",
            "model": job.requested_model or job.workflow,
            "created_at": job.created_at,
            "status": status,
            "progress": progress,
            "seconds": seconds,
            "size": size,
            "quality": quality,
            "url": url,
            "expires_at": expires_at,
            "remixed_from_video_id": None,
            "error": err,
        }

    @app.get("/v1/videos/{video_id}/content")
    async def openai_videos_content(
        request: Request,
        video_id: str,
        authorization: Optional[str] = Header(default=None),
    ) -> Any:
        _require_download_access(cfg, request, authorization)
        job_id = _as_job_id_from_video_id(video_id)
        job = await jobs.get_job(job_id)
        if not job:
            raise _openai_error("Video not found", http_status=404)
        if job.status != "completed":
            raise _openai_error("Video not ready", http_status=409)

        pick = None
        for o in job.outputs or []:
            mt = (o.media_type or "").lower()
            if mt.startswith("video/"):
                pick = o
                break
            fn = (o.filename or "").lower()
            if fn.endswith((".mp4", ".webm", ".mov", ".gif")):
                pick = o
                break
        if pick is None and job.outputs:
            pick = job.outputs[0]
        if pick is None:
            raise _openai_error("No outputs produced", http_status=500)

        path = (cfg.runs_dir / job_id / pick.filename).resolve()
        if not path.exists():
            raise _openai_error("Output file missing", http_status=500)

        return FileResponse(path=str(path), media_type=pick.media_type or None, filename=pick.filename)

    @app.post("/v1/video/generations")
    async def newapi_video_generations_create(
        request: Request,
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _require_auth(cfg, authorization)

        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise _openai_error("Missing 'prompt'")

        model = str(body.get("model") or "").strip()
        duration = body.get("duration")
        seconds = str(duration) if duration is not None else ""
        size = str(body.get("size") or "").strip()

        image_rel = ""
        kind = "txt2video"
        image_val = body.get("image")
        if isinstance(image_val, str) and image_val.strip():
            kind = "img2video"
            image_rel = await _store_input_image_value(image_val, filename_hint="image")

        resolved = await _workflow_from_model_or_default(kind=kind, model=model, has_image=bool(image_rel))
        wf = resolved.workflow
        workflow = wf.name
        requested_model = resolved.record.slug
        standard_params = _collect_standard_params(
            {
                "duration": body.get("duration"),
                "size": body.get("size"),
                "fps": body.get("fps"),
                "frames": body.get("frames"),
                "width": body.get("width"),
                "height": body.get("height"),
                "steps": body.get("steps"),
                "cfg": body.get("cfg"),
                "seed": body.get("seed"),
            }
        )
        standard_params.update(
            await _collect_workflow_request_params(
                spec=wf.parameter_spec,
                values=body,
                store_input_image_bytes=_store_input_image_bytes,
                store_input_image_value=_store_input_image_value,
                skip={
                    *STANDARD_PARAMETER_ORDER,
                    "prompt",
                    "model",
                    "workflow",
                    "duration",
                    "size",
                    "fps",
                    "frames",
                    "width",
                    "height",
                    "steps",
                    "cfg",
                    "seed",
                    "image",
                    "metadata",
                    "response_format",
                },
            )
        )

        meta_obj: Dict[str, Any] = {}
        raw_meta = body.get("metadata")
        if isinstance(raw_meta, dict):
            meta_obj.update(raw_meta)
        for k in ("fps", "seed", "response_format"):
            if k in body:
                meta_obj[k] = body.get(k)

        job = await jobs.create_job(
            kind=kind,
            workflow=workflow,
            platform="New-API",
            requested_model=requested_model,
            seconds=seconds,
            size=size,
            quality="standard",
            metadata=json.dumps(meta_obj, ensure_ascii=False) if meta_obj else "",
            prompt=prompt,
            image=image_rel,
            standard_params=standard_params,
        )

        return {"task_id": job.job_id, "status": "queued"}

    @app.get("/v1/video/generations/{task_id}")
    async def newapi_video_generations_get(
        request: Request,
        task_id: str,
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        _require_auth(cfg, authorization)
        job = await jobs.get_job(task_id)
        if not job:
            raise _openai_error("Task not found", http_status=404)

        status = "queued"
        if job.status == "running":
            status = "in_progress"
        elif job.status == "completed":
            status = "completed"
        elif job.status == "failed":
            status = "failed"

        url = None
        expires_at = None
        fmt = None
        if status == "completed":
            url, expires_at = _authorized_url_parts(request, f"/v1/videos/{_video_id(job.job_id)}/content", authorization)
            for o in job.outputs or []:
                fn = (o.filename or "")
                if "." in fn:
                    fmt = fn.rsplit(".", 1)[-1].lower()
                    break

        meta_out: Any = None
        if (job.metadata or "").strip():
            try:
                meta_out = json.loads(job.metadata)
            except Exception:
                meta_out = job.metadata

        err = None
        if status == "failed":
            err = {"code": 500, "message": job.error or "failed"}

        return {
            "task_id": job.job_id,
            "status": status,
            "url": url,
            "expires_at": expires_at,
            "format": fmt,
            "metadata": meta_out,
            "error": err,
        }

    @app.post("/v1/audio/speech")
    async def openai_audio_speech(
        request: Request,
        body: Dict[str, Any],
        authorization: Optional[str] = Header(default=None),
        x_comfyui_async: Optional[str] = Header(default=None),
    ) -> Any:
        # OpenAI TTS contract: POST {model, input, voice, response_format,
        # speed, instructions} -> audio bytes in the response body. The actual
        # audio container is decided by the workflow's SaveAudio node, so the
        # file's real MIME type is returned and `response_format` is accepted
        # but not enforced (no transcoding dependency).
        _require_auth(cfg, authorization)
        input_text = str(body.get("input") or "").strip()
        if not input_text:
            raise _openai_error("Missing 'input'", http_status=400)
        resolved = await _resolve_public_model(
            model=str(body.get("model") or "").strip(),
            kind="txt2audio",
            has_image=False,
        )
        wf = resolved.workflow
        standard_params = _collect_standard_params(
            {key: body.get(key) for key in AUDIO_PARAMETER_ORDER if key in body}
        )
        # Values are coerced and range-checked against the workflow's sidecar
        # inside create_job, which is the single choke point for every endpoint.
        job = await jobs.create_job(
            kind="txt2audio",
            workflow=wf.name,
            platform="OpenAI",
            requested_model=resolved.record.slug,
            prompt=input_text,
            standard_params=standard_params,
        )
        if x_comfyui_async and str(x_comfyui_async).strip() not in {"0", "false", "False"}:
            return {"job_id": job.job_id, "status": "pending"}

        done = await _openai_wait(job.job_id)
        outputs = [o for o in (done.get("outputs") or []) if isinstance(o, dict)]
        pick = _pick_audio_output(outputs)
        filename = _output_filename(pick) if pick is not None else ""
        if not filename:
            raise _openai_error("Speech generation produced no audio output", http_status=500)
        path = Path(cfg.runs_dir) / str(done.get("job_id") or job.job_id) / filename
        if not path.is_file():
            raise _openai_error("Audio output file missing", http_status=500)
        return FileResponse(
            path=str(path),
            media_type=guess_media_type(filename),
            filename=filename,
        )

    if cfg.ui_enabled:
        from .webui import mount_webui

        mount_webui(app, cfg.ui_dist_dir)

    return app


app = create_app()
