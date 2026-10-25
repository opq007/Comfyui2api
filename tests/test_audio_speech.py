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


class AudioSpeechEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.runs_dir = root / "runs"
        workflows = root / "workflows"
        data = root / "data"
        for folder in (self.runs_dir, workflows, data):
            folder.mkdir(parents=True)
        _write_tts_workflow(workflows / "tts.json")
        _write_tts_voice_workflow(workflows)
        env = {
            "API_TOKEN": "api-token",
            "ADMIN_TOKEN": "admin-token",
            "DATA_DIR": str(data),
            "DATABASE_PATH": str(data / "tasks.db"),
            "ENABLE_WORKFLOW_WATCH": "0",
            "RUNS_DIR": str(self.runs_dir),
            "WORKFLOWS_DIR": str(workflows),
            "SCHEDULER_WAKE_INTERVAL_S": "0.05",
        }
        self._env_patcher = patch.dict(os.environ, env, clear=False)
        self._env_patcher.start()

        import comfyui2api.app as app_module

        self.app = importlib.reload(app_module).create_app()
        from tests.helpers import seed_ready_backend_sync

        self.client_cm = TestClient(self.app)
        self.client = self.client_cm.__enter__()
        seed_ready_backend_sync(self.app, workflows=[("tts-model", "tts.json")], instance_slug="gpu-a")

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)
        self._env_patcher.stop()
        self._tmp.cleanup()

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


if __name__ == "__main__":
    unittest.main()
