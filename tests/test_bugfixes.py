from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


async def _hang_forever(*_args: object, **_kwargs: object) -> None:
    await asyncio.Event().wait()


def _pool_cfg(**overrides: object) -> SimpleNamespace:
    cfg = SimpleNamespace(
        health_check_interval_s=60,
        health_check_timeout_s=1,
        health_check_recovery_threshold=1,
        health_check_fail_threshold=3,
        http_timeout_s=5,
        timeout_s=30,
        scheduler_wake_interval_s=0.05,
        poll_interval_s=0.01,
        runs_dir=None,
        input_subdir="comfyui2api",
        api_token="test-api-token",
        signed_url_secret="",
        signed_url_ttl_seconds=3600,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _instance_record(slug: str = "gpu-a", **overrides: object) -> SimpleNamespace:
    record = SimpleNamespace(
        slug=slug,
        base_url="http://127.0.0.1:8188",
        enabled=True,
        auth_token=None,
        health_interval_s=None,
        max_in_flight=1,
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def _fake_store(*records: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        get_instance=AsyncMock(side_effect=lambda slug: next((r for r in records if r.slug == slug), None)),
        list_instances=AsyncMock(return_value=list(records)),
    )


class InstancePoolProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_loop_probes_before_sleeping(self) -> None:
        # Regression: the loop used to sleep a full health_check_interval_s
        # before its first probe, so an instance registered without probe_now
        # (or recovering after the startup probe missed) stayed UNKNOWN for
        # the whole interval while jobs polled emptily.
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        probe_mock = AsyncMock()
        with patch.object(pool, "_probe_once", probe_mock):
            await pool.sync_instance(record, probe_now=False)
            await asyncio.sleep(0.2)
        self.assertGreaterEqual(probe_mock.await_count, 1)
        await pool.aclose()

    async def test_probe_loop_uses_fast_interval_when_not_healthy(self) -> None:
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        probe_mock = AsyncMock()
        with (
            patch.object(pool, "_probe_once", probe_mock),
            patch.object(pool_module, "PROBE_FAST_INTERVAL_S", 0.05),
        ):
            await pool.sync_instance(record, probe_now=False)
            await asyncio.sleep(0.3)
        self.assertGreaterEqual(probe_mock.await_count, 3)
        await pool.aclose()

    async def test_probe_loop_survives_store_error(self) -> None:
        # Regression: a transient store failure used to terminate the probe
        # task silently and freeze the health state forever.
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        store = _fake_store(record)
        calls = {"n": 0}

        async def _get_instance(slug: str) -> object:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("db locked")
            return record

        store.get_instance = AsyncMock(side_effect=_get_instance)
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=store)
        probe_mock = AsyncMock()
        with (
            patch.object(pool, "_probe_once", probe_mock),
            patch.object(pool_module, "PROBE_FAST_INTERVAL_S", 0.05),
        ):
            await pool.sync_instance(record, probe_now=False)
            await asyncio.sleep(0.2)
        self.assertGreaterEqual(probe_mock.await_count, 1)
        assert pool._runtimes[record.slug].probe_task is not None
        self.assertFalse(pool._runtimes[record.slug].probe_task.done())
        await pool.aclose()

    async def test_probe_once_failure_path_updates_state(self) -> None:
        # Guards the failure branch of _probe_once (consecutive_failures /
        # UNKNOWN transition) after the warm-object-info refactor.
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        await pool.sync_instance(record, probe_now=False)
        async with pool._lock:
            runtime = pool._runtimes[record.slug]
        runtime.health_client = SimpleNamespace(system_stats=AsyncMock(side_effect=RuntimeError("backend down")), aclose=AsyncMock())
        await pool._probe_once(record.slug, runtime.generation)
        self.assertEqual(runtime.consecutive_failures, 1)
        self.assertEqual(runtime.health, pool_module.HEALTH_UNKNOWN)
        self.assertIn("backend down", str(runtime.last_error))
        await pool.aclose()

    async def test_probe_once_success_warms_object_info(self) -> None:
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        await pool.sync_instance(record, probe_now=False)
        warm_client = SimpleNamespace(
            object_info=AsyncMock(return_value={"nodes": {}}),
            object_info_cached=False,
            aclose=AsyncMock(),
        )
        async with pool._lock:
            runtime = pool._runtimes[record.slug]
            runtime.job_client = warm_client
            runtime.health_client = SimpleNamespace(system_stats=AsyncMock(return_value={}), aclose=AsyncMock())
        await pool._probe_once(record.slug, runtime.generation)
        await asyncio.sleep(0.2)
        self.assertEqual(runtime.health, pool_module.HEALTH_HEALTHY)
        warm_client.object_info.assert_awaited_once()
        await pool.aclose()

    async def test_kick_probes_throttled_per_slug(self) -> None:
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        probe_mock = AsyncMock()
        async with pool._lock:
            pool._runtimes[record.slug] = pool_module.InstanceRuntime(slug=record.slug)
        with (
            patch.object(pool, "_probe_once", probe_mock),
            patch.object(pool_module, "KICK_THROTTLE_S", 0.05),
        ):
            await pool.kick_probes([record.slug])
            await pool.kick_probes([record.slug])  # inside throttle window
            await asyncio.sleep(0.15)
            self.assertEqual(probe_mock.await_count, 1)
            await asyncio.sleep(0.05)  # window elapsed
            await pool.kick_probes([record.slug])
            await asyncio.sleep(0.15)
            self.assertEqual(probe_mock.await_count, 2)

    async def test_choose_and_reserve_does_not_oversubscribe(self) -> None:
        # Regression: readiness check and in_flight increment used to run in
        # two separate lock acquisitions, letting concurrent dispatches
        # exceed max_in_flight.
        import comfyui2api.instance_pool as pool_module

        record = _instance_record(max_in_flight=1)
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        async with pool._lock:
            runtime = pool._runtimes.get(record.slug)
            if runtime is None:
                runtime = pool_module.InstanceRuntime(slug=record.slug)
                pool._runtimes[record.slug] = runtime
            runtime.health = pool_module.HEALTH_HEALTHY
        results = await asyncio.gather(
            pool.choose_and_reserve(model_slug="m", policy="round_robin", pool_slugs=[record.slug]),
            pool.choose_and_reserve(model_slug="m", policy="round_robin", pool_slugs=[record.slug]),
        )
        self.assertEqual(sorted(item for item in results if item), [record.slug])
        self.assertIn(None, results)
        async with pool._lock:
            self.assertEqual(pool._runtimes[record.slug].in_flight, 1)

    async def test_sync_instance_defers_job_client_close_until_idle(self) -> None:
        # Regression: recreating clients closed the old job_client while
        # in-flight jobs were still using it, killing them mid-run.
        import comfyui2api.instance_pool as pool_module

        record = _instance_record()
        pool = pool_module.InstancePool(cfg=_pool_cfg(), store=_fake_store(record))
        old_client = SimpleNamespace(
            aclose=AsyncMock(),
            object_info_cached=True,
        )
        with patch.object(pool, "_probe_once", AsyncMock()):
            await pool.sync_instance(record, probe_now=False)
            async with pool._lock:
                runtime = pool._runtimes[record.slug]
                runtime.job_client = old_client
                runtime.in_flight = 1
            updated = _instance_record(base_url="http://127.0.0.1:9999")
            with patch.object(pool_module, "DELAYED_CLOSE_POLL_S", 0.05):
                await pool.sync_instance(updated, probe_now=False)
                await asyncio.sleep(0.2)
                old_client.aclose.assert_not_awaited()  # still in flight
                async with pool._lock:
                    pool._runtimes[record.slug].in_flight = 0
                await asyncio.sleep(0.3)
        old_client.aclose.assert_awaited_once()
        await pool.aclose()


def _scheduler_manager(*, record: SimpleNamespace, job: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=_pool_cfg(),
        backend=SimpleNamespace(get_model=AsyncMock(return_value=record)),
        pool=SimpleNamespace(
            choose_and_reserve=AsyncMock(return_value="gpu-a"),
            client_for=AsyncMock(),
            release=AsyncMock(),
            ready_slugs=AsyncMock(return_value=["gpu-a"]),
            kick_probes=AsyncMock(),
            wait_for_slot=AsyncMock(return_value=True),
        ),
        registry=SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(name="w.json")),
            list=AsyncMock(return_value=[]),
        ),
        get_job=AsyncMock(return_value=job),
        _update=AsyncMock(),
        fail_job=AsyncMock(),
        _run_job=AsyncMock(),
    )


def _fake_job() -> SimpleNamespace:
    from comfyui2api.util import utc_now_unix

    return SimpleNamespace(
        job_id="job-1",
        created_at=utc_now_unix(),
        model_slug="model-a",
        workflow="w.json",
        status="pending",
        prompt_id="",
        image="",
        standard_params={},
        kind="txt2img",
    )


class JobSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scan_once_kicks_probes_when_blocked(self) -> None:
        from comfyui2api.job_scheduler import JobScheduler

        record = SimpleNamespace(
            slug="model-a",
            enabled=True,
            workflow_name="w.json",
            routing_policy="round_robin",
            instance_slugs=("gpu-a", "gpu-b"),
        )
        manager = _scheduler_manager(record=record, job=_fake_job())
        manager.pool.ready_slugs = AsyncMock(return_value=[])
        scheduler = JobScheduler(manager)
        scheduler.enqueue("job-1")
        dispatched = await scheduler._scan_once()
        self.assertFalse(dispatched)
        manager.pool.kick_probes.assert_awaited_once_with(["gpu-a", "gpu-b"])
        self.assertEqual(list(scheduler._queued), ["job-1"])

    async def test_scheduler_loop_survives_scan_exception(self) -> None:
        # Regression: one exception inside _scan_once used to kill the
        # scheduler task silently; every queued job then stayed pending
        # forever (no timeout failure either, since timeout is only checked
        # inside the scan).
        from comfyui2api.job_scheduler import JobScheduler

        record = SimpleNamespace(
            slug="model-a",
            enabled=True,
            workflow_name="w.json",
            routing_policy="round_robin",
            instance_slugs=("gpu-a",),
        )
        manager = _scheduler_manager(record=record, job=_fake_job())
        manager.get_job = AsyncMock(side_effect=[RuntimeError("db locked"), None])
        scheduler = JobScheduler(manager)
        await scheduler.start()
        try:
            scheduler.enqueue("job-1")
            await asyncio.sleep(0.3)
            assert scheduler._task is not None
            self.assertFalse(scheduler._task.done())
            self.assertEqual(list(scheduler._queued), [])  # second pass removed the ghost job
        finally:
            await scheduler.stop()

    async def test_dispatch_reroutes_on_httpx_transport_error(self) -> None:
        # Regression: httpx.ConnectError/ReadTimeout are neither OSError nor
        # builtin TimeoutError, so the reroute path never fired for the most
        # common failure (backend just went down).
        from comfyui2api.job_scheduler import JobScheduler

        record = SimpleNamespace(
            slug="model-a",
            enabled=True,
            workflow_name="w.json",
            routing_policy="round_robin",
            instance_slugs=("gpu-a", "gpu-b"),
        )
        manager = _scheduler_manager(record=record, job=_fake_job())
        manager.pool.choose_and_reserve = AsyncMock(side_effect=["gpu-a", "gpu-b"])
        manager.pool.release = AsyncMock()
        manager._run_job = AsyncMock(side_effect=[httpx.ConnectError("connection refused"), None])
        scheduler = JobScheduler(manager)
        await scheduler._dispatch("job-1")
        self.assertEqual(manager._run_job.await_count, 2)
        manager.fail_job.assert_not_awaited()
        self.assertEqual(manager.pool.release.await_count, 2)

    async def test_dispatch_cancels_queued_comfy_prompt_on_late_failure(self) -> None:
        from comfyui2api.comfy_client import ComfyApiError
        from comfyui2api.job_scheduler import JobScheduler

        record = SimpleNamespace(
            slug="model-a",
            enabled=True,
            workflow_name="w.json",
            routing_policy="round_robin",
            instance_slugs=("gpu-a",),
        )
        job = _fake_job()
        job.prompt_id = "prompt-1"
        manager = _scheduler_manager(record=record, job=job)
        client = SimpleNamespace(delete_queue_entries=AsyncMock())
        manager.pool.client_for = AsyncMock(return_value=client)
        manager._run_job = AsyncMock(side_effect=ComfyApiError("execution failed"))
        scheduler = JobScheduler(manager)
        await scheduler._dispatch("job-1")
        client.delete_queue_entries.assert_awaited_once_with(["prompt-1"])
        manager.fail_job.assert_awaited_once()


class RunJobTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def _manager(self, client: SimpleNamespace, timeout_s: int) -> object:
        import comfyui2api.jobs as jobs_module
        from tests.helpers import fake_job_deps

        deps = fake_job_deps(client=client, timeout_s=timeout_s, runs_dir=self._tmp_runs)
        return jobs_module.JobManager(
            cfg=deps.cfg,
            registry=SimpleNamespace(),
            pool=deps.pool,
            backend=deps.backend,
        )

    def _workflow(self) -> SimpleNamespace:
        return SimpleNamespace(
            name="test.json",
            parameter_spec=None,
            clone_obj=lambda: {
                "prompt": {
                    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                    "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
                }
            },
        )

    async def test_run_job_times_out_when_ws_and_history_never_complete(self) -> None:
        # Regression: the WS monitor had no timeout, so a ComfyUI process
        # alive at the transport layer but never executing parked the job in
        # "queued" forever (and leaked its instance slot).
        import comfyui2api.jobs as jobs_module
        from comfyui2api.comfy_client import QueuedPrompt

        with tempfile.TemporaryDirectory() as tmp:
            self._tmp_runs = Path(tmp)
            client = SimpleNamespace(
                object_info=AsyncMock(return_value={}),
                queue_prompt=AsyncMock(return_value=QueuedPrompt(prompt_id="prompt-1", client_id="client-1", number=1)),
                get_history_entry=AsyncMock(return_value=None),
                wait_for_history_complete=AsyncMock(side_effect=_hang_forever),
                view_bytes=AsyncMock(return_value=b""),
            )
            manager = self._manager(client, timeout_s=1)
            workflow = self._workflow()
            job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")
            with (
                patch.object(manager, "_resolve_workflow", AsyncMock(return_value=workflow)),
                patch.object(jobs_module, "prepare_prompt", return_value=({"1": {}}, None, [], {"positive": [], "negative": []})),
                patch.object(manager, "_monitor_ws", AsyncMock(side_effect=_hang_forever)),
            ):
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(manager._run_job(job.job_id, client=client), timeout=6)

    async def test_run_job_cancellation_propagates_without_history_fallback(self) -> None:
        # Regression: `except BaseException` swallowed CancelledError and
        # fell into the long history poll, stalling shutdown and leaking the
        # ws task.
        import comfyui2api.jobs as jobs_module
        from comfyui2api.comfy_client import QueuedPrompt

        with tempfile.TemporaryDirectory() as tmp:
            self._tmp_runs = Path(tmp)
            client = SimpleNamespace(
                object_info=AsyncMock(return_value={}),
                queue_prompt=AsyncMock(return_value=QueuedPrompt(prompt_id="prompt-1", client_id="client-1", number=1)),
                get_history_entry=AsyncMock(return_value=None),
                wait_for_history_complete=AsyncMock(side_effect=_hang_forever),
                view_bytes=AsyncMock(return_value=b""),
            )
            manager = self._manager(client, timeout_s=30)
            workflow = self._workflow()
            job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")
            with (
                patch.object(manager, "_resolve_workflow", AsyncMock(return_value=workflow)),
                patch.object(jobs_module, "prepare_prompt", return_value=({"1": {}}, None, [], {"positive": [], "negative": []})),
                patch.object(manager, "_monitor_ws", AsyncMock(side_effect=_hang_forever)),
            ):
                task = asyncio.create_task(manager._run_job(job.job_id, client=client))
                await asyncio.sleep(0.2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            client.wait_for_history_complete.assert_not_awaited()


class BackendStoreGuardTests(unittest.TestCase):
    def test_delete_instance_blocked_while_job_running(self) -> None:
        import comfyui2api.backend_store as backend_module
        import comfyui2api.job_store as job_store_module
        from comfyui2api.errors import ConflictError

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "data.db"
            backend = backend_module.BackendStore(db)
            jobs_store = job_store_module.JobStore(db)
            asyncio.run(self._seed_and_assert(backend, jobs_store))

    @staticmethod
    async def _seed_and_assert(backend: object, jobs_store: object) -> None:
        from comfyui2api.backend_store import BackendStore
        from comfyui2api.errors import ConflictError
        from comfyui2api.jobs import Job

        assert isinstance(backend, BackendStore)
        await backend.init()
        await jobs_store.init()
        try:
            record = await backend.create_instance({"slug": "gpu-a", "base_url": "http://127.0.0.1:8188", "enabled": True})
            job = Job(
                job_id="j1",
                created_at_utc="2026-01-01T00:00:00Z",
                created_at=1,
                status="running",
                kind="txt2img",
                workflow="w.json",
                instance_slug="gpu-a",
            )
            await jobs_store.upsert_job(job)
            try:
                await backend.delete_instance("gpu-a")
            except ConflictError:
                pass
            else:
                raise AssertionError("delete_instance must refuse while a job is running on the instance")
            # Sanity: guard passes once the job is terminal.
            job.status = "completed"
            await jobs_store.upsert_job(job)
            await backend.delete_instance("gpu-a")
        finally:
            # BackendStore opens one short-lived connection per op (no aclose);
            # JobStore keeps a persistent connection that must be closed before
            # the temp dir is removed on Windows.
            await jobs_store.aclose()


class PublicJobWsAuthFrameTests(unittest.TestCase):
    def test_unauthorized_ws_gets_json_error_frame_then_close(self) -> None:
        # Regression: the public job WS used to close without accepting, so
        # uvicorn answered the handshake with HTTP 403 and browsers only saw
        # a generic transport error.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir()
            (root / "runs").mkdir()
            (root / "data").mkdir()
            env = {
                "API_TOKEN": "api-token",
                "ADMIN_TOKEN": "admin-token",
                "DATA_DIR": str(root / "data"),
                "DATABASE_PATH": str(root / "data" / "tasks.db"),
                "ENABLE_WORKFLOW_WATCH": "0",
                "RUNS_DIR": str(root / "runs"),
                "WORKFLOWS_DIR": str(root / "workflows"),
            }
            patcher = patch.dict(os.environ, env, clear=False)
            patcher.start()
            import comfyui2api.app as app_module

            app = importlib.reload(app_module).create_app()
            self.addCleanup(patcher.stop)
            with TestClient(app) as client:
                with client.websocket_connect("/v1/jobs/whatever/ws") as ws:
                    frame = ws.receive_json()
                self.assertEqual(frame["type"], "error")
                self.assertEqual(frame["data"]["status"], 401)


def _verify_signature(url: str, secret: str) -> bool:
    from urllib.parse import parse_qs, urlparse

    from comfyui2api.signed_urls import has_valid_signature

    parsed = urlparse(url)
    flat = {
        key: (value[0] if isinstance(value, list) and value else value)
        for key, value in parse_qs(parsed.query).items()
    }
    return bool(has_valid_signature(path=parsed.path, query_params=flat, secret=secret))


class SignedMediaUrlEventsTests(unittest.IsolatedAsyncioTestCase):
    """WS-pushed events must carry signed /runs URLs: browsers cannot attach
    Authorization headers to <img>/<video> tags, so bare paths 401 at the
    download endpoint and playground results show as broken images."""

    def _manager(self, runs_dir: Path) -> object:
        import comfyui2api.jobs as jobs_module
        from tests.helpers import fake_job_deps

        deps = fake_job_deps(runs_dir=runs_dir)
        return jobs_module.JobManager(
            cfg=deps.cfg,
            registry=SimpleNamespace(),
            pool=deps.pool,
            backend=deps.backend,
        )

    def _secret_for(self, manager: object) -> str:
        from comfyui2api.signed_urls import signing_secret

        cfg = manager.cfg  # type: ignore[attr-defined]
        return signing_secret(configured_secret=cfg.signed_url_secret, api_token=cfg.api_token)

    async def test_public_job_signs_runs_urls(self) -> None:
        from comfyui2api.jobs import JobOutput

        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(Path(tmp))
            job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")
            await manager._update(
                job.job_id,
                status="completed",
                url="/runs/j1/preview.png",
                outputs=[
                    JobOutput(
                        filename="preview.png",
                        url="/runs/j1/preview.png",
                        media_type="image/png",
                        node_id="2",
                        output_key="images",
                    )
                ],
            )
            public = manager.public_job(job)
            for url in (public["url"], public["outputs"][0]["url"]):
                self.assertTrue(url.startswith("/runs/j1/preview.png?"))
                self.assertIn("sig=", url)
                self.assertTrue(_verify_signature(url, self._secret_for(manager)))
            # The stored job keeps the bare path; signing is presentation-only.
            self.assertEqual(job.url, "/runs/j1/preview.png")
            # Non-/runs URLs pass through untouched.
            from comfyui2api.jobs import _signed_media_url

            self.assertEqual(_signed_media_url(manager.cfg, ""), "")
            self.assertEqual(_signed_media_url(manager.cfg, "https://example.com/a.png"), "https://example.com/a.png")

    async def test_job_completed_ws_event_carries_signed_url(self) -> None:
        import comfyui2api.jobs as jobs_module
        from comfyui2api.comfy_client import QueuedPrompt

        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(Path(tmp))
            client = SimpleNamespace(
                object_info=AsyncMock(return_value={}),
                queue_prompt=AsyncMock(return_value=QueuedPrompt(prompt_id="prompt-1", client_id="client-1", number=1)),
                get_history_entry=AsyncMock(return_value=None),
                wait_for_history_complete=AsyncMock(
                    return_value={
                        "outputs": {
                            "save-node": {
                                "images": [
                                    {"filename": "result.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        }
                    }
                ),
                view_bytes=AsyncMock(return_value=b"png-bytes"),
            )
            workflow = SimpleNamespace(
                name="test.json",
                parameter_spec=None,
                clone_obj=lambda: {
                    "prompt": {
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
                    }
                },
            )
            class _FakeWs:
                def __init__(self) -> None:
                    self.frames: list[dict] = []

                async def send_json(self, payload: dict) -> None:
                    self.frames.append(payload)

            fake_ws = _FakeWs()
            job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")
            await manager.subscribe(job.job_id, fake_ws)
            with (
                patch.object(manager, "_resolve_workflow", AsyncMock(return_value=workflow)),
                patch.object(jobs_module, "prepare_prompt", return_value=({"1": {}}, None, [], {"positive": [], "negative": []})),
                patch.object(manager, "_monitor_ws", AsyncMock(return_value=None)),
            ):
                await manager._run_job(job.job_id, client=client)

            completed_frames = [f for f in fake_ws.frames if f.get("type") == "job_completed"]
            self.assertEqual(len(completed_frames), 1)
            data = completed_frames[0]["data"]
            self.assertIn("sig=", data["url"])
            self.assertTrue(_verify_signature(data["url"], self._secret_for(manager)))
            for output in data["outputs"]:
                self.assertIn("sig=", output["url"])
                self.assertTrue(_verify_signature(output["url"], self._secret_for(manager)))


class WsSnapshotUrlDownloadTests(unittest.TestCase):
    def test_ws_snapshot_url_downloads_without_auth_header(self) -> None:
        # End to end: connect to the public job WS, take the URL from the
        # snapshot frame, fetch it with no Authorization header (exactly what
        # an <img> tag does) and expect the media bytes back.
        from comfyui2api.jobs import Job, JobOutput

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            workflows_dir = root / "workflows"
            data_dir = root / "data"
            runs_dir.mkdir(parents=True)
            workflows_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            (runs_dir / "job-e2e").mkdir()
            (runs_dir / "job-e2e" / "2__preview.png").write_bytes(b"png-bytes-e2e")

            env = {
                "API_TOKEN": "secret-token",
                "ADMIN_TOKEN": "admin-token",
                "DATA_DIR": str(data_dir),
                "DATABASE_PATH": str(data_dir / "tasks.db"),
                "ENABLE_WORKFLOW_WATCH": "0",
                "RUNS_DIR": str(runs_dir),
                "WORKFLOWS_DIR": str(workflows_dir),
            }
            patcher = patch.dict(os.environ, env, clear=False)
            patcher.start()
            import comfyui2api.app as app_module

            app = importlib.reload(app_module).create_app()
            try:
                job = Job(
                    job_id="job-e2e",
                    created_at_utc="2026-03-19T00:00:00Z",
                    created_at=123,
                    status="completed",
                    kind="txt2img",
                    workflow="demo.json",
                    url="/runs/job-e2e/2__preview.png",
                    outputs=[
                        JobOutput(
                            filename="2__preview.png",
                            url="/runs/job-e2e/2__preview.png",
                            media_type="image/png",
                            node_id="2",
                            output_key="images",
                        )
                    ],
                )
                with patch.object(app.state.jobs, "get_job", AsyncMock(return_value=job)):
                    with TestClient(app) as client:
                        with client.websocket_connect("/v1/jobs/job-e2e/ws?api_key=secret-token") as ws:
                            snapshot = ws.receive_json()
                        self.assertEqual(snapshot["type"], "job_snapshot")
                        url = snapshot["data"]["url"]
                        self.assertIn("sig=", url)
                        response = client.get(url)  # no Authorization header
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.content, b"png-bytes-e2e")
            finally:
                patcher.stop()


if __name__ == "__main__":
    unittest.main()
