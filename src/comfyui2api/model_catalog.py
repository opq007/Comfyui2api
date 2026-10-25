from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend_store import BackendStore, ExternalModelRecord
from .errors import ValidationError
from .instance_pool import InstancePool
from .workflow_registry import WorkflowDefinition, WorkflowRegistry


@dataclass(frozen=True, slots=True)
class PublicModel:
    slug: str
    display_name: str | None
    created_at: int
    kinds: tuple[str, ...]
    ready: bool
    workflow_available: bool

    def as_openai(self) -> dict[str, Any]:
        return {
            "id": self.slug,
            "object": "model",
            "created": self.created_at,
            "owned_by": "comfyui2api",
            "display_name": self.display_name or self.slug,
            "kind": list(self.kinds),
            "ready": self.ready,
            "workflow_available": self.workflow_available,
        }


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    record: ExternalModelRecord
    workflow: WorkflowDefinition
    kinds: tuple[str, ...]
    ready: bool
    kind: str


class ModelNotFoundError(LookupError):
    def __init__(self, slug: str) -> None:
        super().__init__(f"The model `{slug}` does not exist.")
        self.slug = slug
        self.code = "model_not_found"


class WorkflowUnavailableError(ValueError):
    def __init__(self, slug: str) -> None:
        super().__init__(f"Workflow bound to model `{slug}` is unavailable.")
        self.slug = slug
        self.code = "workflow_unavailable"


class NoAvailableBackendError(RuntimeError):
    def __init__(self, slug: str) -> None:
        super().__init__(f"No healthy ComfyUI instance available for model `{slug}`.")
        self.slug = slug
        self.code = "no_available_backend"


class KindMismatchError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelCatalog:
    def __init__(self, *, store: BackendStore, pool: InstancePool, registry: WorkflowRegistry) -> None:
        self.store = store
        self.pool = pool
        self.registry = registry

    async def list_public(self) -> list[PublicModel]:
        models = await self.store.list_models(enabled_only=True)
        out: list[PublicModel] = []
        for record in models:
            workflow, kinds, available = await self.workflow_meta(record.workflow_name)
            healthy = await self.pool.healthy_slugs(list(record.instance_slugs))
            out.append(
                PublicModel(
                    slug=record.slug,
                    display_name=record.display_name,
                    created_at=record.created_at,
                    kinds=kinds,
                    ready=available and bool(healthy),
                    workflow_available=available,
                )
            )
        return out

    async def resolve(
        self,
        *,
        slug: str,
        kind: str | None,
        has_image: bool,
        require_healthy: bool = True,
    ) -> ResolvedModel:
        requested = (slug or "").strip()
        if not requested:
            raise ModelNotFoundError("")
        record = await self.store.get_model(requested)
        if record is None or not record.enabled:
            raise ModelNotFoundError(requested)
        workflow, kinds, available = await self.workflow_meta(record.workflow_name)
        if not available or workflow is None:
            raise WorkflowUnavailableError(record.slug)
        chosen = _choose_kind(
            kind=kind,
            has_image=has_image,
            kinds=kinds,
            workflow=workflow,
        )
        if require_healthy:
            healthy = await self.pool.healthy_slugs(list(record.instance_slugs))
            if not healthy:
                raise NoAvailableBackendError(record.slug)
        return ResolvedModel(
            record=record,
            workflow=workflow,
            kinds=kinds,
            ready=True,
            kind=chosen,
        )

    async def assert_enableable(self, workflow_name: str | None) -> None:
        name = (workflow_name or "").strip()
        if not name:
            raise ValidationError("workflow_name is required to enable a model")
        workflow = await self._get_workflow(name)
        if workflow is None:
            raise ValidationError(f"Workflow '{name}' is unavailable")

    async def workflow_meta(self, workflow_name: str | None) -> tuple[WorkflowDefinition | None, tuple[str, ...], bool]:
        name = (workflow_name or "").strip()
        if not name:
            return None, (), False
        workflow = await self._get_workflow(name)
        if workflow is None:
            return None, (), False
        return workflow, capability_kinds(workflow.capabilities), True

    async def _get_workflow(self, name: str) -> WorkflowDefinition | None:
        workflow = await self.registry.get(name)
        if workflow is not None:
            return workflow
        requested = name.lower()
        for item in await self.registry.list():
            if item.name.lower() == requested:
                return item
        return None


def capability_kinds(caps: Any) -> tuple[str, ...]:
    kinds: list[str] = []
    has_load = bool(getattr(caps, "has_load_image", False))
    if bool(getattr(caps, "has_save_image", False)):
        kinds.append("txt2img")
        if has_load:
            kinds.append("img2img")
    if bool(getattr(caps, "has_save_video", False)):
        kinds.append("txt2video")
        if has_load:
            kinds.append("img2video")
    if bool(getattr(caps, "has_save_audio", False)):
        kinds.append("txt2audio")
    return tuple(kinds)


def _choose_kind(*, kind: str | None, has_image: bool, kinds: tuple[str, ...], workflow: WorkflowDefinition) -> str:
    requested = (kind or "").strip()
    if requested:
        if requested not in kinds:
            raise KindMismatchError(_kind_error(workflow, requested))
        return requested
    prefix = "img" if has_image else "txt"
    matches = [item for item in kinds if item.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        needed = "img2img" if has_image else "txt2img"
        raise KindMismatchError(_kind_error(workflow, needed))
    raise KindMismatchError(
        f"Workflow '{workflow.name}' supports multiple kinds {matches}; pass 'kind' explicitly."
    )


def _kind_error(workflow: WorkflowDefinition, kind: str) -> str:
    caps = workflow.capabilities
    missing: list[str] = []
    if kind in {"img2img", "img2video"} and not getattr(caps, "has_load_image", False):
        missing.append("missing LoadImage")
    if kind in {"txt2img", "img2img"} and not getattr(caps, "has_save_image", False):
        missing.append("missing SaveImage")
    if kind in {"txt2video", "img2video"} and not getattr(caps, "has_save_video", False):
        missing.append("missing SaveVideo")
    if kind == "txt2audio" and not getattr(caps, "has_save_audio", False):
        missing.append("missing SaveAudio")
    detail = f"detected kind={getattr(caps, 'kind', 'unknown')}"
    if missing:
        detail += f"; {', '.join(missing)}"
    return f"Workflow '{workflow.name}' does not support {kind} ({detail})."
