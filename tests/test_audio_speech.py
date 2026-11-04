from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


WAV_BYTES = b"RIFF-fake-wav-bytes"


def _write_tts_workflow(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt": {
                    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                    "2": {"class_type": "SaveAudio", "inputs": {"filename_prefix": "tts"}},
                }
            }
        ),
        encoding="utf-8",
    )


def _write_tts_voice_workflow(workflows: Path) -> None:
    (workflows / "tts_voice.json").write_text(
        json.dumps(
            {
                "prompt": {
                    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                    "3": {"class_type": "TTSVoice", "inputs": {"voice": "default", "speed": "1.0"}},
                    "2": {"class_type": "SaveAudio", "inputs": {"filename_prefix": "tts"}},
                }
            }
        ),
        encoding="utf-8",
    )
    sidecar_dir = workflows / ".comfyui2api"
    sidecar_dir.mkdir(exist_ok=True)
    (sidecar_dir / "tts_voice.params.json").write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "txt2audio",
                "prompt_node": "1.text",
                "parameters": {
                    "voice": {
                        "type": "string",
                        "description": "TTS voice name",
                        "maps": [{"target": "3.voice"}],
                    },
                    "speed": {
                        "type": "string",
                        "description": "TTS speed",
                        "maps": [{"target": "3.speed"}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _audio_history(prompt_id: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "status": {"status_str": "success", "completed": True},
        "outputs": {
            "2": {
                "audio": [
                    {"filename": "tts_00001_.wav", "subfolder": "", "type": "output"}
                ]
            }
        },
    }


def _fake_speech_client(prompt_id: str = "prompt-tts-1") -> SimpleNamespace:
    from comfyui2api.comfy_client import QueuedPrompt

    async def _empty_ws(*_args: object, **_kwargs: object):
        return
        yield  # pragma: no cover - makes this an async generator

    return SimpleNamespace(
        object_info=AsyncMock(return_value={}),
        queue_prompt=AsyncMock(return_value=QueuedPrompt(prompt_id=prompt_id, client_id="client-1", number=1)),
        ws_events=_empty_ws,
        get_history_entry=AsyncMock(return_value=_audio_history(prompt_id)),
        wait_for_history_complete=AsyncMock(return_value=_audio_history(prompt_id)),
        view_bytes=AsyncMock(return_value=WAV_BYTES),
        delete_queue_entries=AsyncMock(),
        aclose=AsyncMock(),
    )


class AudioCapabilityTests(unittest.TestCase):
    def test_saveaudio_workflow_detected_as_txt2audio(self) -> None:
        from comfyui2api.comfy_workflow import detect_capabilities
        from comfyui2api.model_catalog import capability_kinds

        prompt = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi"}},
            "2": {"class_type": "SaveAudio", "inputs": {"filename_prefix": "tts"}},
        }
        caps = detect_capabilities(prompt)
        self.assertTrue(caps.has_save_audio)
        self.assertEqual(caps.kind, "txt2audio")
        self.assertEqual(capability_kinds(caps), ("txt2audio",))

    def test_custom_saveaudio_node_names_detected(self) -> None:
        from comfyui2api.comfy_workflow import detect_capabilities

        for cls in ("SaveAudio", "SaveAudioMP3", "Impact_SaveAudio"):
            caps = detect_capabilities({"1": {"class_type": cls, "inputs": {}}})
            self.assertEqual(caps.kind, "txt2audio", cls)


class _SpeechAppTestCase(unittest.TestCase):
    """Temp dirs + env + TestClient shared by the TTS endpoint suites."""

    def _check_available(self) -> None:
        """Override to skip before any temp directory is created."""

    def _prepare_workflows(self, root: Path) -> Path:
        workflows = root / "workflows"
        workflows.mkdir(parents=True)
        return workflows

    def _seed_models(self) -> None:
        """Override to create the external model rows this suite needs."""

    def setUp(self) -> None:
        self._check_available()
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs_dir = root / "runs"
        data = root / "data"
        for folder in (self.runs_dir, data):
            folder.mkdir(parents=True)
        self.workflows_dir = self._prepare_workflows(root)
        env = {
            "API_TOKEN": "api-token",
            "ADMIN_TOKEN": "admin-token",
            "DATA_DIR": str(data),
            "DATABASE_PATH": str(data / "tasks.db"),
            "ENABLE_WORKFLOW_WATCH": "0",
            "RUNS_DIR": str(self.runs_dir),
            "WORKFLOWS_DIR": str(self.workflows_dir),
            "SCHEDULER_WAKE_INTERVAL_S": "0.05",
        }
        self._env_patcher = patch.dict(os.environ, env, clear=False)
        self._env_patcher.start()

        import comfyui2api.app as app_module

        self.app = importlib.reload(app_module).create_app()
        self.client_cm = TestClient(self.app)
        self.client = self.client_cm.__enter__()
        self._seed_models()

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)
        self._env_patcher.stop()
        self._tmp.cleanup()


class AudioSpeechEndpointTests(_SpeechAppTestCase):
    def _prepare_workflows(self, root: Path) -> Path:
        workflows = super()._prepare_workflows(root)
        _write_tts_workflow(workflows / "tts.json")
        _write_tts_voice_workflow(workflows)
        return workflows

    def _seed_models(self) -> None:
        from tests.helpers import seed_ready_backend_sync

        seed_ready_backend_sync(self.app, workflows=[("tts-model", "tts.json")], instance_slug="gpu-a")

    def test_speech_sync_returns_audio_bytes_with_real_mime(self) -> None:
        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        fake_client = _fake_speech_client()
        with patch.object(self.app.state.pool, "client_for", AsyncMock(return_value=fake_client)):
            response = self.client.post(
                "/v1/audio/speech",
                headers={"Authorization": "Bearer api-token"},
                json={
                    "model": "tts-model",
                    "input": "你好，世界",
                    "voice": "alloy",
                    "speed": 1.2,
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"].split(";")[0], "audio/wav")
        self.assertEqual(response.content, WAV_BYTES)

        # voice/speed must be passed through as standard params (bindable by sidecars)
        manager = self.app.state.jobs
        import asyncio

        job_records = asyncio.run(manager.list_jobs(limit=10))
        tts_jobs = [j for j in job_records if j.kind == "txt2audio"]
        self.assertEqual(len(tts_jobs), 1)
        self.assertEqual(tts_jobs[0].standard_params.get("voice"), "alloy")
        self.assertEqual(float(tts_jobs[0].standard_params.get("speed")), 1.2)

    def test_speech_async_header_returns_job_id(self) -> None:
        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        response = self.client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer api-token", "x-comfyui-async": "1"},
            json={"model": "tts-model", "input": "async hello"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("job_id", payload)
        self.assertEqual(payload["status"], "pending")

    def test_speech_missing_input_is_400(self) -> None:
        response = self.client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer api-token"},
            json={"model": "tts-model"},
        )
        self.assertEqual(response.status_code, 400)

    def test_speech_model_shows_txt2audio_kind_in_models_list(self) -> None:
        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        response = self.client.get("/v1/models", headers={"Authorization": "Bearer api-token"})
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        tts = [item for item in data if item["id"] == "tts-model"]
        self.assertEqual(len(tts), 1)
        self.assertIn("txt2audio", tts[0]["kind"])

    def test_speech_unauthorized_is_401(self) -> None:
        response = self.client.post("/v1/audio/speech", json={"model": "tts-model", "input": "x"})
        self.assertEqual(response.status_code, 401)

    def test_speech_unknown_model_is_404(self) -> None:
        response = self.client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer api-token"},
            json={"model": "nope", "input": "x"},
        )
        self.assertEqual(response.status_code, 404)

    def test_speech_picks_audio_output_over_ancillary_files(self) -> None:
        # A TTS workflow may also emit non-audio side files; the endpoint
        # must return the audio one, not whatever lands in outputs[0].
        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        mixed_history = {
            "prompt_id": "prompt-tts-2",
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "9": {"images": [{"filename": "waveform.png", "subfolder": "", "type": "output"}]},
                "2": {"audio": [{"filename": "tts_00001_.wav", "subfolder": "", "type": "output"}]},
            },
        }
        fake_client = _fake_speech_client()
        fake_client.get_history_entry = AsyncMock(return_value=mixed_history)
        fake_client.wait_for_history_complete = AsyncMock(return_value=mixed_history)
        with patch.object(self.app.state.pool, "client_for", AsyncMock(return_value=fake_client)):
            response = self.client.post(
                "/v1/audio/speech",
                headers={"Authorization": "Bearer api-token"},
                json={"model": "tts-model", "input": "mixed outputs"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"].split(";")[0], "audio/wav")
        self.assertEqual(response.content, WAV_BYTES)

    def test_speech_skips_node_pack_panel_entries_it_cannot_fetch(self) -> None:
        # WAS Node Suite adds a compare-panel entry to the history of its audio
        # nodes (hash-named .wav, type="was_compare", served by its own route),
        # and ComfyUI's /view answers 400 "Invalid type" for it. Downloading
        # that entry used to fail the job before the saved mp3 was returned.
        import asyncio

        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        mp3_bytes = b"ID3-fake-mp3-bytes"
        history = {
            "prompt_id": "prompt-tts-3",
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "19": {
                    "audio_sides": [
                        {
                            "filename": "7001648be39185c955473b7fcb6efda8.wav",
                            "subfolder": "",
                            "type": "was_compare",
                        }
                    ]
                },
                "17": {
                    "audio": [
                        {"filename": "ComfyUI_00001_.mp3", "subfolder": "audio", "type": "output"}
                    ]
                },
            },
        }
        fake_client = _fake_speech_client(prompt_id="prompt-tts-3")
        fake_client.get_history_entry = AsyncMock(return_value=history)
        fake_client.wait_for_history_complete = AsyncMock(return_value=history)
        fake_client.view_bytes = AsyncMock(return_value=mp3_bytes)
        with patch.object(self.app.state.pool, "client_for", AsyncMock(return_value=fake_client)):
            response = self.client.post(
                "/v1/audio/speech",
                headers={"Authorization": "Bearer api-token"},
                json={"model": "tts-model", "input": "panel entry"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"].split(";")[0], "audio/mpeg")
        self.assertEqual(response.content, mp3_bytes)
        # Only the saved file was fetched, from its own folder and subfolder.
        fake_client.view_bytes.assert_awaited_once_with(
            filename="ComfyUI_00001_.mp3", subfolder="audio", folder_type="output"
        )
        job = asyncio.run(self.app.state.jobs.list_jobs(limit=10))[0]
        self.assertEqual([output.filename for output in job.outputs], ["17__ComfyUI_00001_.mp3"])
        self.assertEqual(job.outputs[0].media_type, "audio/mpeg")

    def test_speech_sidecar_binds_voice_and_speed_to_node_inputs(self) -> None:
        # Acceptance #3 evidence: a sidecar-declared `voice`/`speed` parameter
        # must be rewritten into the TTS node inputs of the queued prompt.
        import asyncio

        from tests.helpers import mark_instances_healthy_sync, seed_ready_backend_sync

        seed_ready_backend_sync(
            self.app,
            workflows=[("tts-voice-model", "tts_voice.json")],
            instance_slug="gpu-b",
            base_url="http://127.0.0.1:8288",
        )
        mark_instances_healthy_sync(self.app)
        fake_client = _fake_speech_client()
        with patch.object(self.app.state.pool, "client_for", AsyncMock(return_value=fake_client)):
            response = self.client.post(
                "/v1/audio/speech",
                headers={"Authorization": "Bearer api-token"},
                json={"model": "tts-voice-model", "input": "voice test", "voice": "alloy", "speed": "1.2"},
            )
        self.assertEqual(response.status_code, 200, response.text)

        manager = self.app.state.jobs
        job_records = asyncio.run(manager.list_jobs(limit=10))
        job = next(j for j in job_records if j.kind == "txt2audio" and j.workflow == "tts_voice.json")
        prompt_path = Path(job.run_dir) / "prompt.json"
        self.assertTrue(prompt_path.is_file(), f"missing {prompt_path}")
        graph = json.loads(prompt_path.read_text(encoding="utf-8"))["prompt"]
        self.assertEqual(graph["3"]["inputs"]["voice"], "alloy")
        self.assertEqual(graph["3"]["inputs"]["speed"], "1.2")
        self.assertEqual(graph["1"]["inputs"]["text"], "voice test")


class VoxCPM2SidecarTests(unittest.TestCase):
    """The bundled VoxCPM2 workflow must honour OpenAI TTS request params."""

    def _workflow_paths(self) -> tuple[Path, Path]:
        project_root = Path(__file__).resolve().parents[1]
        workflows_dir = project_root / "comfyui-api-workflows"
        workflow_path = workflows_dir / "voxcpm2_base_api.json"
        sidecar_path = workflows_dir / ".comfyui2api" / "voxcpm2_base_api.params.json"
        if not workflow_path.exists() or not sidecar_path.exists():
            self.skipTest(f"Workflow fixture not present: {workflow_path.name}")
        return workflows_dir, workflow_path

    def test_sidecar_loads_as_txt2audio_and_declares_audio_params(self) -> None:
        from comfyui2api.comfy_workflow import detect_capabilities, extract_prompt_and_extra
        from comfyui2api.workflow_params import AUDIO_PARAMETER_ORDER, load_workflow_parameter_spec

        workflows_dir, workflow_path = self._workflow_paths()
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))
        prompt, _extra = extract_prompt_and_extra(workflow_obj)
        self.assertEqual(detect_capabilities(prompt).kind, "txt2audio")

        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="txt2audio",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompt_node, "18.value")
        # Every OpenAI TTS passthrough param must have a binding, otherwise the
        # endpoint silently accepts it and drops it.
        missing = set(AUDIO_PARAMETER_ORDER) - set(spec.parameters)
        self.assertFalse(missing, f"unbound audio params: {sorted(missing)}")
        speed = spec.parameters["speed"]
        self.assertEqual(speed.type, "float")
        self.assertEqual([mapping.ref for mapping in speed.maps], ["19.speed"])
        self.assertEqual(speed.minimum, 0.25)
        self.assertEqual(speed.maximum, 4)

    def test_sidecar_maps_voice_instructions_and_reference_transcript(self) -> None:
        from comfyui2api.workflow_params import load_workflow_parameter_spec, resolve_standard_overrides

        workflows_dir, workflow_path = self._workflow_paths()
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))
        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="txt2audio",
        )
        assert spec is not None

        # No request params -> no overrides, the workflow's own values win.
        self.assertEqual(resolve_standard_overrides(workflow_obj=workflow_obj, spec=spec, request_params={}), [])

        overrides = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={
                "voice": "lyf2",
                "instructions": "warm and calm narration",
                "speed": 1.2,
                "seed": 7,
                "steps": 20,
                "cfg": 3.5,
                "prompt_text": "参考音频的转写",
            },
        )
        self.assertIn(("7", "string", "lyf2"), overrides)
        self.assertIn(("1", "voice_design", "warm and calm narration"), overrides)
        self.assertIn(("19", "speed", 1.2), overrides)
        self.assertIn(("1", "seed", 7), overrides)
        self.assertIn(("1", "steps", 20), overrides)
        self.assertIn(("1", "cfg", 3.5), overrides)
        self.assertIn(("3", "prompt_text", "参考音频的转写"), overrides)

        # OpenAI clients (and the playground) may send speed as a string; the
        # float coercion has to survive that path too.
        as_string = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={"speed": "1.5"},
        )
        self.assertEqual(as_string, [("19", "speed", 1.5)])


class VoxCPM2SpeechEndpointTests(_SpeechAppTestCase):
    """End-to-end OpenAI /v1/audio/speech against the real bundled workflow."""

    def _check_available(self) -> None:
        workflows_dir = Path(__file__).resolve().parents[1] / "comfyui-api-workflows"
        for name in ("voxcpm2_base_api.json", ".comfyui2api/voxcpm2_base_api.params.json"):
            if not (workflows_dir / name).exists():
                self.skipTest(f"Workflow fixture not present: {name}")

    def _prepare_workflows(self, root: Path) -> Path:
        # Serve the repo's real workflows dir read-only.
        return Path(__file__).resolve().parents[1] / "comfyui-api-workflows"

    def _seed_models(self) -> None:
        from tests.helpers import seed_ready_backend_sync

        seed_ready_backend_sync(
            self.app,
            workflows=[("voxcpm2", "voxcpm2_base_api.json")],
            instance_slug="gpu-tts",
        )

    def _latest_job(self) -> object:
        import asyncio

        records = asyncio.run(self.app.state.jobs.list_jobs(limit=10))
        jobs = [record for record in records if record.kind == "txt2audio"]
        self.assertTrue(jobs, "no txt2audio job was created")
        return jobs[0]

    def test_openai_speech_params_reach_voxcpm_nodes(self) -> None:
        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        fake_client = _fake_speech_client()
        with patch.object(self.app.state.pool, "client_for", AsyncMock(return_value=fake_client)):
            response = self.client.post(
                "/v1/audio/speech",
                headers={"Authorization": "Bearer api-token"},
                json={
                    "model": "voxcpm2",
                    "input": "你好，世界",
                    "voice": "lyf2",
                    "speed": 1.2,
                    "instructions": "warm and calm narration",
                    "response_format": "mp3",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"].split(";")[0], "audio/wav")
        self.assertEqual(response.content, WAV_BYTES)

        job = self._latest_job()
        self.assertEqual(job.standard_params.get("voice"), "lyf2")
        # Stored value is the coerced float, not the raw JSON number.
        self.assertEqual(job.standard_params.get("speed"), 1.2)
        self.assertEqual(job.standard_params.get("instructions"), "warm and calm narration")

        prompt_path = Path(job.run_dir) / "prompt.json"
        self.assertTrue(prompt_path.is_file(), f"missing {prompt_path}")
        graph = json.loads(prompt_path.read_text(encoding="utf-8"))["prompt"]
        self.assertEqual(graph["18"]["inputs"]["value"], "你好，世界")
        self.assertEqual(graph["7"]["inputs"]["string"], "lyf2")
        self.assertEqual(graph["1"]["inputs"]["voice_design"], "warm and calm narration")
        # speed lands on the WAS time-stretch node that now feeds SaveAudio.
        self.assertEqual(graph["19"]["inputs"]["speed"], 1.2)
        self.assertEqual(graph["19"]["inputs"]["method"], "keep pitch")
        self.assertEqual(graph["17"]["inputs"]["audio"], ["19", 0])
        # Untouched knobs keep the workflow's own values.
        self.assertEqual(graph["1"]["inputs"]["steps"], 10)
        self.assertEqual(graph["1"]["inputs"]["cfg"], 2)
        self.assertEqual(graph["3"]["inputs"]["prompt_text"], "从什么时候起，愉悦变成了有一点距离感的词。")
        # The sidecar exposes seed, so the job randomizes it per request.
        workflow_seed = json.loads(
            (self.workflows_dir / "voxcpm2_base_api.json").read_text(encoding="utf-8")
        )["1"]["inputs"]["seed"]
        self.assertIsInstance(graph["1"]["inputs"]["seed"], int)
        self.assertNotEqual(graph["1"]["inputs"]["seed"], workflow_seed)

    def test_openai_speech_rejects_invalid_speed_with_400(self) -> None:
        import asyncio

        for speed in (5, 0.1, "fast"):
            with self.subTest(speed=speed):
                response = self.client.post(
                    "/v1/audio/speech",
                    headers={"Authorization": "Bearer api-token"},
                    json={"model": "voxcpm2", "input": "hello", "speed": speed},
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

        # JSON allows a bare NaN literal and httpx's json= encoder refuses to
        # produce one, so send the body raw.
        nan_response = self.client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer api-token", "Content-Type": "application/json"},
            content='{"model": "voxcpm2", "input": "hello", "speed": NaN}',
        )
        self.assertEqual(nan_response.status_code, 400, nan_response.text)
        self.assertEqual(nan_response.json()["error"]["type"], "invalid_request_error")

        # Rejected at the edge: nothing was enqueued.
        self.assertEqual(asyncio.run(self.app.state.jobs.list_jobs(limit=10)), [])

    def test_native_jobs_rejects_invalid_speed_with_400(self) -> None:
        # create_job is the single validation choke point, so the native
        # endpoint rejects the same value the OpenAI one does.
        import asyncio

        response = self.client.post(
            "/v1/jobs",
            headers={"Authorization": "Bearer api-token"},
            json={"model": "voxcpm2", "kind": "txt2audio", "prompt": "x", "speed": 5},
        )
        self.assertEqual(response.status_code, 400, response.text)
        payload = response.json()["error"]
        self.assertEqual(payload["type"], "invalid_request_error")
        self.assertIn("speed", payload["message"])
        self.assertEqual(asyncio.run(self.app.state.jobs.list_jobs(limit=10)), [])

    def test_native_jobs_forwards_prompt_text_and_sampler_knobs(self) -> None:
        response = self.client.post(
            "/v1/jobs",
            headers={"Authorization": "Bearer api-token"},
            json={
                "model": "voxcpm2",
                "kind": "txt2audio",
                "prompt": "第二段文本",
                "prompt_text": "参考音频的转写",
                "steps": 16,
                "cfg": 3,
                "seed": 1234,
                "speed": "1.5",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

        job = self._latest_job()
        self.assertEqual(job.prompt, "第二段文本")
        self.assertEqual(job.standard_params.get("prompt_text"), "参考音频的转写")
        self.assertEqual(job.standard_params.get("steps"), 16)
        self.assertEqual(job.standard_params.get("cfg"), 3.0)
        self.assertEqual(job.standard_params.get("seed"), 1234)
        # String speed is coerced by the same create_job validation path.
        self.assertEqual(job.standard_params.get("speed"), 1.5)

    def test_native_jobs_applies_prompt_text_and_speed_to_the_graph(self) -> None:
        # The native endpoint is the only way to pass prompt_text/steps; prove
        # they reach the nodes, not just the stored params.
        import time

        from tests.helpers import mark_instances_healthy_sync

        mark_instances_healthy_sync(self.app)
        fake_client = _fake_speech_client(prompt_id="prompt-native-1")
        with patch.object(self.app.state.pool, "client_for", AsyncMock(return_value=fake_client)):
            response = self.client.post(
                "/v1/jobs",
                headers={"Authorization": "Bearer api-token"},
                json={
                    "model": "voxcpm2",
                    "kind": "txt2audio",
                    "prompt": "第三段文本",
                    "prompt_text": "新的参考转写",
                    "speed": "1.4",
                    "steps": 12,
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["job"]["job_id"]
            status: dict = {}
            deadline = time.time() + 10
            while time.time() < deadline:
                payload = self.client.get(
                    f"/v1/jobs/{job_id}", headers={"Authorization": "Bearer api-token"}
                ).json()
                status = payload.get("job") or payload
                if status.get("status") in {"completed", "failed"}:
                    break
                time.sleep(0.02)

        self.assertEqual(status.get("status"), "completed", status)
        job = self._latest_job()
        prompt_path = Path(job.run_dir) / "prompt.json"
        self.assertTrue(prompt_path.is_file(), f"missing {prompt_path}")
        graph = json.loads(prompt_path.read_text(encoding="utf-8"))["prompt"]
        self.assertEqual(graph["18"]["inputs"]["value"], "第三段文本")
        self.assertEqual(graph["19"]["inputs"]["speed"], 1.4)
        self.assertEqual(graph["3"]["inputs"]["prompt_text"], "新的参考转写")
        self.assertEqual(graph["1"]["inputs"]["steps"], 12)


if __name__ == "__main__":
    unittest.main()
