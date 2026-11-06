from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import comfyui2api.job_store as job_store_module
from comfyui2api.job_store import JobStore
from comfyui2api.jobs import Job, JobOutput, JobManager
from comfyui2api.workflow_params import WorkflowParameterDefinition, WorkflowParameterSpec, WorkflowParamTarget


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.helpers import fake_job_deps


def _completed_job(job_id: str, *, prompt: str, outputs: list[JobOutput] | None = None) -> Job:
    return Job(
        job_id=job_id,
        created_at=1780000000,
        created_at_utc="2026-05-29T00:19:20Z",
        status="completed",
        kind="txt2img",
        workflow="wf.json",
        prompt=prompt,
        progress_percent=100,
        outputs=outputs or [],
        url=(outputs[0].url if outputs else ""),
    )


class FullPromptStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_prompt_persists_untruncated(self) -> None:
        long_prompt = "a" * 2000
        long_negative = "b" * 1500
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "tasks.db")
            await store.init()
            job = _completed_job("task_long", prompt=long_prompt)
            job.negative_prompt = long_negative
            await store.upsert_job(job)

            detail = await store.get_task("task_long")
            self.assertIsNotNone(detail)
            assert detail is not None
            self.assertEqual(detail["task"]["prompt_full"], long_prompt)
            self.assertEqual(detail["task"]["negative_prompt_full"], long_negative)
            # preview stays a bounded display snippet
            self.assertLessEqual(len(detail["task"]["prompt_preview"]), 300 + len("...[truncated]"))
            self.assertTrue(detail["task"]["prompt_preview"].endswith("...[truncated]"))

            listed = await store.list_tasks(q=long_prompt[-40:])
            self.assertEqual(listed["total"], 1)
            self.assertEqual(listed["items"][0]["prompt_full"], long_prompt)
            await store.aclose()


class HasOutputsFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_has_outputs_filters_and_exposes_media_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "tasks.db")
            await store.init()
            with_output = _completed_job(
                "task_with_output",
                prompt="cat",
                outputs=[
                    JobOutput(
                        filename="out.png",
                        url="/runs/task_with_output/out.png",
                        media_type="image/png",
                        node_id="9",
                        output_key="images",
                    )
                ],
            )
            without_output = _completed_job("task_empty", prompt="dog")
            await store.upsert_job(with_output)
            await store.replace_outputs("task_with_output", with_output.outputs)
            await store.upsert_job(without_output)

            listed = await store.list_tasks(has_outputs=True)
            self.assertEqual(listed["total"], 1)
            self.assertEqual(listed["items"][0]["job_id"], "task_with_output")
            self.assertEqual(listed["items"][0]["media_type"], "image/png")

            unfiltered = await store.list_tasks()
            self.assertEqual(unfiltered["total"], 2)
            self.assertNotIn("media_type", unfiltered["items"][1])
            await store.aclose()


@contextmanager
def _retention(limit: int):
    with patch.object(job_store_module, "SUBMISSION_RETENTION", limit):
        yield


class PlaygroundSubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_submission_ring_buffer_enrichment_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "tasks.db")
            await store.init()
            job = _completed_job(
                "task_hist",
                prompt="cat",
                outputs=[
                    JobOutput(
                        filename="out.png",
                        url="/runs/task_hist/out.png",
                        media_type="image/png",
                        node_id="9",
                        output_key="images",
                    )
                ],
            )
            await store.upsert_job(job)
            await store.replace_outputs(job.job_id, job.outputs)

            with _retention(2):
                await store.add_submission(kind="txt2img", model="m-1", params={"prompt": "one"}, job_id="task_old")
                await store.add_submission(kind="txt2img", model="m-1", params={"prompt": "two"}, job_id="task_hist")
                await store.add_submission(kind="txt2video", model="m-2", params={"prompt": "three"}, job_id="missing")

            listed = await store.list_submissions()
            self.assertEqual(listed["total"], 2)  # oldest pruned by the ring buffer
            newest = listed["items"][0]
            self.assertEqual(newest["params"]["prompt"], "three")
            self.assertIsNone(newest["status"])  # linked task no longer exists
            enriched = listed["items"][1]
            self.assertEqual(enriched["job_id"], "task_hist")
            self.assertEqual(enriched["status"], "completed")
            self.assertEqual(enriched["result_url"], "/runs/task_hist/out.png")
            self.assertEqual(enriched["media_type"], "image/png")

            cleared = await store.clear_submissions()
            self.assertEqual(cleared["deleted"], 2)
            empty = await store.list_submissions()
            self.assertEqual(empty["total"], 0)
            await store.aclose()

    async def test_submission_rejects_missing_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "tasks.db")
            await store.init()
            with self.assertRaises(ValueError):
                await store.add_submission(kind="txt2img", model="m", params={}, job_id="")
            await store.aclose()


def _seed_spec() -> WorkflowParameterSpec:
    return WorkflowParameterSpec(
        version=1,
        kind="txt2img",
        parameters={
            "seed": WorkflowParameterDefinition(
                name="seed",
                type="int",
                maps=(WorkflowParamTarget(ref="3.seed"),),
            )
        },
        path=Path("seed.params.json"),
        prompt_node="6.text",
    )


def _workflow_with_seed_spec() -> SimpleNamespace:
    return SimpleNamespace(
        name="test.json",
        parameter_spec=_seed_spec(),
        clone_obj=lambda: {
            "prompt": {
                "3": {"class_type": "KSampler", "inputs": {"seed": 0}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
            }
        },
    )


async def _empty_ws_events(client_id: str = ""):
    return
    yield  # pragma: no cover - makes this an async generator


class EffectiveSeedWriteBackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._runs_dir = Path(self._tmp.name)

    def _client(self) -> SimpleNamespace:
        return SimpleNamespace(
            object_info=AsyncMock(return_value={}),
            queue_prompt=AsyncMock(return_value=SimpleNamespace(prompt_id="pid-1", number=1)),
            ws_events=_empty_ws_events,
            get_history_entry=AsyncMock(
                return_value={"outputs": {}, "status": {"completed": True, "status_str": "success"}}
            ),
            wait_for_history_complete=AsyncMock(return_value={"outputs": {}}),
            view_bytes=AsyncMock(return_value=b""),
        )

    def _manager(self, client: SimpleNamespace, store: JobStore) -> JobManager:
        deps = fake_job_deps(client=client, runs_dir=self._runs_dir)
        return JobManager(
            cfg=deps.cfg,
            registry=SimpleNamespace(get=AsyncMock(return_value=_workflow_with_seed_spec()), list=AsyncMock(return_value=[])),
            pool=deps.pool,
            backend=deps.backend,
            store=store,
        )

    async def test_randomized_seed_is_persisted_as_effective_param(self) -> None:
        with tempfile.TemporaryDirectory() as db_tmp:
            store = JobStore(Path(db_tmp) / "tasks.db")
            await store.init()
            client = self._client()
            manager = self._manager(client, store)
            job = await manager.create_job(kind="txt2img", workflow="test.json", prompt="hello")
            self.assertNotIn("seed", job.standard_params)

            await manager._run_job(job.job_id, client=client)

            detail = await store.get_task(job.job_id)
            self.assertIsNotNone(detail)
            assert detail is not None
            effective_seed = detail["task"]["request_json"]["standard_params"]["seed"]
            self.assertIsInstance(effective_seed, int)
            self.assertGreaterEqual(effective_seed, 0)
            self.assertLess(effective_seed, 2**32)
            # prompt survives untruncated in the request summary
            self.assertEqual(detail["task"]["request_json"]["prompt"], "hello")
            await manager.stop_workers()
            await store.aclose()

    async def test_explicit_seed_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as db_tmp:
            store = JobStore(Path(db_tmp) / "tasks.db")
            await store.init()
            client = self._client()
            manager = self._manager(client, store)
            job = await manager.create_job(
                kind="txt2img", workflow="test.json", prompt="hello", standard_params={"seed": 12345}
            )
            await manager._run_job(job.job_id, client=client)

            detail = await store.get_task(job.job_id)
            assert detail is not None
            self.assertEqual(detail["task"]["request_json"]["standard_params"]["seed"], 12345)
            await manager.stop_workers()
            await store.aclose()


if __name__ == "__main__":
    unittest.main()
