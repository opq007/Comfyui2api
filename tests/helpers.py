from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock


async def seed_ready_backend(
    app: Any,
    *,
    workflows: list[tuple[str, str]],
    instance_slug: str = "gpu-a",
    base_url: str = "http://127.0.0.1:8188",
) -> None:
    backend = app.state.backend
    pool = app.state.pool
    record = await backend.create_instance(
        {
            "slug": instance_slug,
            "base_url": base_url,
            "enabled": True,
        }
    )
    await pool.sync_instance(record, probe_now=False)
    async with pool._lock:
        runtime = pool._runtimes[instance_slug]
        runtime.health = "healthy"
        runtime.consecutive_successes = 1
        runtime.consecutive_failures = 0
    for slug, workflow_name in workflows:
        await backend.create_model(
            {
                "slug": slug,
                "workflow_name": workflow_name,
                "enabled": True,
                "instance_slugs": [instance_slug],
            }
        )


def seed_ready_backend_sync(app: Any, **kwargs: Any) -> None:
    asyncio.run(seed_ready_backend(app, **kwargs))


def mark_instances_healthy_sync(app: Any) -> None:
    asyncio.run(mark_instances_healthy(app))


async def mark_instances_healthy(app: Any) -> None:
    pool = app.state.pool
    instances = await app.state.backend.list_instances()
    async with pool._lock:
        for record in instances:
            runtime = pool._runtimes.get(record.slug)
            if runtime is None:
                continue
            runtime.health = "healthy"
            runtime.consecutive_successes = 1
            runtime.consecutive_failures = 0


def fake_job_deps(*, client: Any | None = None, timeout_s: int = 30, poll_interval_s: float = 0.01, runs_dir: Any = None) -> SimpleNamespace:
    job_client = client or SimpleNamespace(
        object_info=AsyncMock(return_value={}),
        queue_prompt=AsyncMock(),
        wait_for_history_complete=AsyncMock(return_value={"outputs": {}}),
        view_bytes=AsyncMock(return_value=b""),
        ws_events=AsyncMock(),
    )
    pool = SimpleNamespace(
        choose_and_reserve=AsyncMock(return_value="gpu-a"),
        client_for=AsyncMock(return_value=job_client),
        release=AsyncMock(),
        ready_slugs=AsyncMock(return_value=["gpu-a"]),
        wait_for_slot=AsyncMock(return_value=True),
    )
    backend = SimpleNamespace(
        get_model=AsyncMock(
            return_value=SimpleNamespace(
                slug="demo",
                enabled=True,
                workflow_name="test.json",
                routing_policy="round_robin",
                instance_slugs=("gpu-a",),
            )
        )
    )
    cfg = SimpleNamespace(
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        runs_dir=runs_dir,
        input_subdir="comfyui2api",
        api_token="test-api-token",
        signed_url_secret="",
        signed_url_ttl_seconds=3600,
    )
    return SimpleNamespace(cfg=cfg, pool=pool, backend=backend, client=job_client)
