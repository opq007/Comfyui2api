from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.helpers import fake_job_deps

if TYPE_CHECKING:
    from comfyui2api.jobs import Job, JobManager


COMPLETE_HISTORY = {
    "outputs": {},
    "status": {"status_str": "success", "completed": True, "messages": []},
}


async def _hang_events(**_kwargs):
    """A WebSocket that stays open and never delivers a message.

    This is the shape of the reported failure: ComfyUI's push channel goes
    quiet (events sent before the monitor attached, a restarted backend, a
    cleared queue) while the socket itself keeps answering pings.
    """
    await asyncio.Event().wait()
    yield {}  # pragma: no cover - never reached


class JobCompletionWatchdogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp_runs = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _manager(self, client: SimpleNamespace, *, timeout_s: int = 30) -> "JobManager":
        import comfyui2api.jobs as jobs_module

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

    async def _run(self, manager: "JobManager", client: SimpleNamespace) -> "Job":
        import comfyui2api.jobs as jobs_module
        from comfyui2api.comfy_client import QueuedPrompt

        workflow = self._workflow()
        job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")
        client.object_info = AsyncMock(return_value={})
        client.queue_prompt = AsyncMock(
            return_value=QueuedPrompt(prompt_id="prompt-1", client_id="client-1", number=1)
        )
        with (
            patch.object(manager, "_resolve_workflow", AsyncMock(return_value=workflow)),
            patch.object(
                jobs_module,
                "prepare_prompt",
                return_value=({"1": {}}, None, [], {"positive": [], "negative": []}),
            ),
        ):
            await manager._run_job(job.job_id, client=client)
        return job

    def _client(self, **overrides: object) -> SimpleNamespace:
        client = SimpleNamespace(
            object_info=AsyncMock(return_value={}),
            queue_prompt=AsyncMock(),
            ws_events=_hang_events,
            get_history_entry=AsyncMock(return_value=None),
            get_queue=AsyncMock(return_value={"queue_running": [], "queue_pending": []}),
            wait_for_history_complete=AsyncMock(side_effect=AssertionError("history poll must not be the fallback")),
            view_bytes=AsyncMock(return_value=b""),
        )
        for key, value in overrides.items():
            setattr(client, key, value)
        return client

    async def test_completes_from_history_while_ws_stays_silent(self) -> None:
        # The reported bug: the push channel goes quiet, so the job used to sit
        # in "queued" (0%, instance slot reserved) until TIMEOUT_S even though
        # ComfyUI had already finished the prompt.
        import comfyui2api.jobs as jobs_module

        client = self._client(get_history_entry=AsyncMock(return_value=COMPLETE_HISTORY))
        manager = self._manager(client, timeout_s=30)
        with patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")
        client.wait_for_history_complete.assert_not_awaited()

    async def test_ws_traffic_skips_the_history_probe(self) -> None:
        # The probe must not run while push events are still arriving — that is
        # the "no redundant poller" property the old 0.5s poller was removed for.
        # Three watch slices pass with events flowing, so a non-gated watchdog
        # would have probed twice before the terminal event.
        import comfyui2api.jobs as jobs_module

        async def _chatty_events(**_kwargs):
            for _ in range(30):
                await asyncio.sleep(0.02)
                yield {"type": "progress", "data": {"value": 1, "max": 99}}
            yield {"type": "execution_success", "data": {"prompt_id": "prompt-1"}}

        client = self._client(
            ws_events=_chatty_events,
            get_history_entry=AsyncMock(return_value=COMPLETE_HISTORY),
        )
        manager = self._manager(client, timeout_s=30)
        with patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.2):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")
        client.get_queue.assert_not_awaited()
        # exactly one: the post-terminal history read, never a watchdog probe.
        self.assertEqual(client.get_history_entry.await_count, 1)

    async def test_lost_executing_event_is_recovered_from_comfyui_queue(self) -> None:
        # A dropped `executing` event must not leave the task at "queued" 0%
        # for its whole run: the watchdog reports what ComfyUI is doing.
        import comfyui2api.jobs as jobs_module

        running = {"queue_running": [[0, "prompt-1", {}, {}, []]], "queue_pending": []}
        statuses: list[str] = []

        client = self._client(
            get_history_entry=AsyncMock(side_effect=[None, None, COMPLETE_HISTORY, COMPLETE_HISTORY]),
            get_queue=AsyncMock(return_value=running),
        )
        manager = self._manager(client, timeout_s=30)
        original_update = manager._update

        async def _spy(job_id: str, **fields: object) -> object:
            if "status" in fields:
                statuses.append(str(fields["status"]))
            return await original_update(job_id, **fields)

        with (
            patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02),
            patch.object(jobs_module, "WS_SILENCE_GRACE_S", 0.0),
            patch.object(manager, "_update", _spy),
        ):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertIn("running", statuses)
        self.assertEqual(job.status, "completed")

    async def test_vanished_prompt_fails_instead_of_waiting_out_the_timeout(self) -> None:
        # ComfyUI restarted / queue cleared: there is nothing to wait for, so
        # the slot must be freed with a real error instead of a 1h "queued".
        import comfyui2api.jobs as jobs_module
        from comfyui2api.comfy_client import ComfyApiError

        client = self._client()
        manager = self._manager(client, timeout_s=30)
        with (
            patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02),
            patch.object(jobs_module, "WS_SILENCE_GRACE_S", 0.0),
        ):
            with self.assertRaises(ComfyApiError) as ctx:
                await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertIn("no longer has this prompt", str(ctx.exception))

    async def test_pending_prompt_is_not_mistaken_for_a_lost_one(self) -> None:
        # The prompt is still waiting its turn on ComfyUI: keep waiting instead
        # of failing a healthy job.
        import comfyui2api.jobs as jobs_module

        pending = {"queue_running": [], "queue_pending": [[0, "prompt-1", {}, {}, []]]}

        client = self._client(
            get_history_entry=AsyncMock(side_effect=[None, None, None, COMPLETE_HISTORY, COMPLETE_HISTORY]),
            get_queue=AsyncMock(return_value=pending),
        )
        manager = self._manager(client, timeout_s=30)
        with (
            patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02),
            patch.object(jobs_module, "WS_SILENCE_GRACE_S", 0.0),
        ):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")

    async def test_unrecognised_queue_payload_does_not_fail_a_healthy_job(self) -> None:
        # Another ComfyUI flavour (or a proxy) answering /queue in an unexpected
        # shape must never be read as "the prompt is gone".
        import comfyui2api.jobs as jobs_module

        client = self._client(
            get_history_entry=AsyncMock(side_effect=[None, None, None, COMPLETE_HISTORY, COMPLETE_HISTORY]),
            get_queue=AsyncMock(return_value={"items": [], "total": 0}),
        )
        manager = self._manager(client, timeout_s=30)
        with (
            patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02),
            patch.object(jobs_module, "WS_SILENCE_GRACE_S", 0.0),
        ):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")

    async def test_foreign_ws_broadcasts_do_not_reset_the_silence_clock(self) -> None:
        # ComfyUI broadcasts `status` to every socket whenever the queue changes.
        # If that counted as activity, a stalled job would look alive forever as
        # long as any other client (the ComfyUI web UI, another in-flight job)
        # touched the instance — and the watchdog would never notice completion.
        import comfyui2api.jobs as jobs_module

        async def _status_only(**_kwargs):
            while True:
                await asyncio.sleep(0.01)
                yield {"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 3}}}}

        client = self._client(
            ws_events=_status_only,
            get_history_entry=AsyncMock(return_value=COMPLETE_HISTORY),
        )
        manager = self._manager(client, timeout_s=30)
        with patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.05):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")
        client.get_history_entry.assert_awaited()

    async def test_unreadable_history_cannot_fail_a_job(self) -> None:
        # /queue says the prompt is gone but /history errors: that is not evidence
        # of a lost prompt, and failing would throw away the outputs of a job that
        # actually finished. Keep waiting until history answers.
        import comfyui2api.jobs as jobs_module

        calls = {"n": 0}

        async def _history(*_args: object, **_kwargs: object) -> object:
            calls["n"] += 1
            if calls["n"] <= 8:  # comfortably past PROMPT_MISSING_CHECKS
                raise OSError("history endpoint unavailable")
            return COMPLETE_HISTORY

        client = self._client(get_history_entry=AsyncMock(side_effect=_history))
        manager = self._manager(client, timeout_s=30)
        with (
            patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02),
            patch.object(jobs_module, "WS_SILENCE_GRACE_S", 0.0),
        ):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")

    async def test_timeout_is_not_paid_twice(self) -> None:
        # The old code waited TIMEOUT_S on the WS and then handed the same
        # budget to the history poll, so a dead backend cost 2x the timeout.
        import time

        import comfyui2api.jobs as jobs_module

        pending = {"queue_running": [], "queue_pending": [[0, "prompt-1", {}, {}, []]]}

        client = self._client(
            get_queue=AsyncMock(return_value=pending),
            wait_for_history_complete=AsyncMock(side_effect=AssertionError("no fallback")),
        )
        manager = self._manager(client, timeout_s=1)
        with patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.05):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(self._run(manager, client), timeout=4)
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        client.wait_for_history_complete.assert_not_awaited()

    async def test_ws_failure_still_falls_back_to_history_polling(self) -> None:
        # A monitor that dies (transport error) must keep the old recovery path.
        import comfyui2api.jobs as jobs_module

        async def _broken_events(**_kwargs):
            yield {"type": "status", "data": {}}
            raise ConnectionResetError("socket dropped")

        client = self._client(
            ws_events=_broken_events,
            get_history_entry=AsyncMock(return_value=None),
            wait_for_history_complete=AsyncMock(return_value=COMPLETE_HISTORY),
        )
        manager = self._manager(client, timeout_s=30)
        with patch.object(jobs_module, "HISTORY_WATCH_INTERVAL_S", 0.02):
            job = await asyncio.wait_for(self._run(manager, client), timeout=5)
        self.assertEqual(job.status, "completed")
        client.wait_for_history_complete.assert_awaited_once()

    async def test_ws_end_fallback_gets_only_the_remaining_budget(self) -> None:
        # A socket that dies late must not restart the clock: the history
        # fallback inherits what is left of TIMEOUT_S instead of a fresh budget.
        # With TIMEOUT_S=10 and the socket dying at ~2s, the handover budget must
        # be strictly below 10 (the old code always passed the full TIMEOUT_S).
        # The margins are >=1s on both sides of the assertion, not milliseconds.
        async def _late_break(**_kwargs):
            yield {"type": "status", "data": {}}
            await asyncio.sleep(2.0)
            raise ConnectionResetError("socket dropped late")

        async def _never(*_args: object, **_kwargs: object) -> object:
            await asyncio.Event().wait()

        client = self._client(
            ws_events=_late_break,
            get_history_entry=AsyncMock(return_value=None),
            wait_for_history_complete=AsyncMock(side_effect=_never),
        )
        manager = self._manager(client, timeout_s=10)
        task = asyncio.create_task(self._run(manager, client))
        await asyncio.sleep(2.6)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNotNone(
            client.wait_for_history_complete.await_args,
            "the monitor died late, the history fallback should have taken over",
        )
        budget = client.wait_for_history_complete.await_args.kwargs["timeout_s"]
        self.assertLessEqual(budget, 9)


if __name__ == "__main__":
    unittest.main()
