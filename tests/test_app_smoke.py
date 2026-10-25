from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        workflows_dir = root / "workflows"
        runs_dir = root / "runs"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        (workflows_dir / ".comfyui2api").mkdir(parents=True, exist_ok=True)

        workflow_name = "test_txt2img.json"
        (workflows_dir / workflow_name).write_text(
            json.dumps(
                {
                    "prompt": {
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
                        "10": {
                            "class_type": "EmptyLatentImage",
                            "inputs": {"width": 512, "height": 512},
                            "_meta": {"title": "Latent Size"},
                        },
                        "11": {
                            "class_type": "KSampler",
                            "inputs": {"seed": 1, "steps": 20, "cfg": 3.5},
                            "_meta": {"title": "Sampler"},
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (workflows_dir / ".comfyui2api" / "test_txt2img.params.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "txt2img",
                    "parameters": {
                        "size": {
                            "type": "size",
                            "maps": [
                                {"target": "10.width", "part": "width"},
                                {"target": "10.height", "part": "height"},
                            ],
                        },
                        "steps": {
                            "type": "int",
                            "default": 20,
                            "maps": [{"target": "11.steps"}],
                        },
                        "cfg": {
                            "type": "float",
                            "default": 3.5,
                            "maps": [{"target": "11.cfg"}],
                        },
                        "seed": {
                            "type": "int",
                            "maps": [{"target": "11.seed"}],
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        txt2video_workflow_name = "test_txt2video.json"
        (workflows_dir / txt2video_workflow_name).write_text(
            json.dumps(
                {
                    "prompt": {
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "2": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "sample"}},
                        "3": {"class_type": "VideoCombine", "inputs": {"fps": 24, "frames": 96}},
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        hybrid_video_workflow_name = "test_hybrid_video.json"
        (workflows_dir / hybrid_video_workflow_name).write_text(
            json.dumps(
                {
                    "prompt": {
                        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "2": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
                        "3": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 24, "images": ["1", 0]}},
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        dual_input_video_workflow_name = "test_dual_input_video.json"
        (workflows_dir / dual_input_video_workflow_name).write_text(
            json.dumps(
                {
                    "prompt": {
                        "325": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "second prompt"}},
                        "437": {"class_type": "LoadImage", "inputs": {"image": "primary.png"}},
                        "438": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "first prompt"}},
                        "440": {"class_type": "LoadImage", "inputs": {"image": "secondary.png"}},
                        "500": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "sample"}},
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (workflows_dir / ".comfyui2api" / "test_dual_input_video.params.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "img2video",
                    "prompt_node": "438.value",
                    "image_node": "437.image",
                    "parameters": {
                        "prompt2": {"type": "string", "maps": [{"target": "325.value"}]},
                        "image2": {"type": "image", "maps": [{"target": "440.image"}]},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        img2img_workflow_name = "test_img2img.json"
        (workflows_dir / img2img_workflow_name).write_text(
            json.dumps(
                {
                    "prompt": {
                        "76": {"class_type": "LoadImage", "inputs": {"image": "primary.png"}},
                        "81": {"class_type": "LoadImage", "inputs": {"image": "secondary.png"}},
                        "94": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sample"}},
                        "125": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
                        "129": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
                        "135": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
                        "137": {"class_type": "KSampler", "inputs": {"steps": 4, "cfg": 1}},
                        "705": {
                            "class_type": "INTConstant",
                            "inputs": {"value": 1},
                            "_meta": {"title": "inputImgCount"},
                        },
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (workflows_dir / ".comfyui2api" / "test_img2img.params.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "img2img",
                    "prompt_node": "135.text",
                    "image_node": "76.image",
                    "parameters": {
                        "n": {"type": "int", "minimum": 1, "maps": [{"target": "129.batch_size"}]},
                        "steps": {"type": "int", "maps": [{"target": "137.steps"}]},
                        "cfg": {"type": "float", "maps": [{"target": "137.cfg"}]},
                        "seed": {"type": "int", "maps": [{"target": "125.noise_seed"}]},
                        "image2": {"type": "image", "maps": [{"target": "81.image"}]},
                        "inputImgCount": {
                            "type": "int",
                            "default": 1,
                            "minimum": 1,
                            "maps": [{"target": "705.value"}],
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        env = {
            "API_TOKEN": "secret-token",
            "ADMIN_TOKEN": "admin-token",
            "DATA_DIR": str(root / "data"),
            "DATABASE_PATH": str(root / "data" / "comfyui2api.db"),
            "ENABLE_WORKFLOW_WATCH": "0",
            "MAX_BODY_BYTES": "1024",
            "RUNS_DIR": str(runs_dir),
            "WORKFLOWS_DIR": str(workflows_dir),
        }
        cls.env_patcher = patch.dict(os.environ, env, clear=False)
        cls.env_patcher.start()

        import comfyui2api.app as app_module

        cls.app_module = importlib.reload(app_module)
        cls.app = cls.app_module.app
        cls.workflow_name = workflow_name
        cls.txt2video_workflow_name = txt2video_workflow_name
        cls.hybrid_video_workflow_name = hybrid_video_workflow_name
        cls.dual_input_video_workflow_name = dual_input_video_workflow_name
        cls.img2img_workflow_name = img2img_workflow_name

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env_patcher.stop()
        cls.tempdir.cleanup()

    def setUp(self) -> None:
        self.client_cm = TestClient(self.app)
        self.client = self.client_cm.__enter__()
        self._seed_backend_via_admin()

    def tearDown(self) -> None:
        self.client_cm.__exit__(None, None, None)

    def _seed_backend_via_admin(self) -> None:
        listed = self.client.get("/v1/admin/models", headers={"Authorization": "Bearer admin-token"})
        if listed.status_code == 200 and listed.json().get("items"):
            self._mark_pool_healthy()
            return
        created = self.client.post(
            "/v1/admin/instances",
            headers={"Authorization": "Bearer admin-token"},
            json={"slug": "gpu-a", "base_url": "http://127.0.0.1:8188"},
        )
        self.assertIn(created.status_code, {200, 409})
        for slug, workflow_name in (
            ("test_txt2img", self.workflow_name),
            ("test_txt2video", self.txt2video_workflow_name),
            ("test_hybrid_video", self.hybrid_video_workflow_name),
            ("test_dual_input_video", self.dual_input_video_workflow_name),
            ("test_img2img", self.img2img_workflow_name),
        ):
            self.client.post(
                "/v1/admin/models",
                headers={"Authorization": "Bearer admin-token"},
                json={
                    "slug": slug,
                    "workflow_name": workflow_name,
                    "enabled": True,
                    "instance_slugs": ["gpu-a"],
                },
            )
        self._mark_pool_healthy()

    def _mark_pool_healthy(self) -> None:
        for runtime in self.app.state.pool._runtimes.values():
            runtime.health = "healthy"
            runtime.consecutive_successes = 1
            runtime.consecutive_failures = 0

    def test_models_require_auth_and_list_loaded_workflow(self) -> None:
        unauthorized = self.client.get("/v1/models")
        self.assertEqual(unauthorized.status_code, 401)

        authorized = self.client.get("/v1/models", headers={"Authorization": "Bearer secret-token"})
        self.assertEqual(authorized.status_code, 200)
        payload = authorized.json()
        self.assertEqual(payload["object"], "list")
        by_id = {item["id"]: item for item in payload["data"]}
        self.assertIn("test_txt2img", by_id)
        self.assertNotIn(self.workflow_name, by_id)
        self.assertEqual(by_id["test_txt2img"]["kind"], ["txt2img"])
        self.assertTrue(by_id["test_txt2img"]["ready"])

    def test_request_body_limit_returns_413(self) -> None:
        response = self.client.post("/v1/images/generations", json={"prompt": "x" * 2048})
        self.assertEqual(response.status_code, 413)
        payload = response.json()
        self.assertIn("Request body too large", payload["error"]["message"])

    def test_images_generations_accepts_model_id_without_json_suffix(self) -> None:
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(job_id="job-images", requested_model="test_txt2img", created_at=123)
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "true"},
                json={"prompt": "cat", "model": "test_txt2img"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "pending")
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["workflow"], self.workflow_name)
        self.assertEqual(kwargs["requested_model"], "test_txt2img")

    def test_chat_completions_routes_text_prompt_to_txt2img(self) -> None:
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(job_id="job-chat-image", requested_model="test_txt2img", created_at=123)
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "true"},
                json={
                    "model": "test_txt2img",
                    "messages": [{"role": "user", "content": "draw a cinematic cat"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "test_txt2img")
        message = payload["choices"][0]["message"]
        self.assertEqual(message["role"], "assistant")
        content = json.loads(message["content"])
        self.assertEqual(content["type"], "generation_job")
        self.assertEqual(content["kind"], "txt2img")
        self.assertEqual(content["status"], "pending")
        self.assertEqual(content["job_id"], "job-chat-image")
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["kind"], "txt2img")
        self.assertEqual(kwargs["workflow"], self.workflow_name)
        self.assertEqual(kwargs["prompt"], "draw a cinematic cat")

    def test_chat_completions_routes_multimodal_prompt_to_img2video(self) -> None:
        image_url = "data:image/png;base64," + base64.b64encode(b"fake-image").decode("ascii")
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(job_id="job-chat-video", requested_model="test_hybrid_video", created_at=123)
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "true"},
                json={
                    "model": "test_hybrid_video",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "animate this frame"},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    "fps": 24,
                    "seconds": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        content = json.loads(payload["choices"][0]["message"]["content"])
        self.assertEqual(content["kind"], "img2video")
        self.assertEqual(content["status"], "pending")
        self.assertEqual(content["video_id"], "video_job-chat-video")
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["kind"], "img2video")
        self.assertEqual(kwargs["workflow"], self.hybrid_video_workflow_name)
        self.assertEqual(kwargs["requested_model"], "test_hybrid_video")
        self.assertEqual(kwargs["prompt"], "animate this frame")
        self.assertTrue(kwargs["image"])
        self.assertEqual(kwargs["standard_params"], {"fps": 24, "duration": 5})

    def test_workflow_parameters_endpoint_exposes_sidecar_mapping(self) -> None:
        response = self.client.get(
            f"/v1/admin/workflows/{self.workflow_name}/parameters",
            headers={"Authorization": "Bearer admin-token"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["workflow"]["name"], self.workflow_name)
        self.assertIsNone(payload["parameter_error"])
        names = [item["name"] for item in payload["parameter_mapping"]["parameters"]]
        self.assertEqual(names[:4], ["size", "steps", "cfg", "seed"])
        detected = payload["detected_candidates"]
        self.assertEqual(detected["size"][0]["maps"][0]["ref"], "10.width")
        self.assertEqual(detected["size"][0]["maps"][1]["ref"], "10.height")
        self.assertEqual(detected["steps"][0]["maps"][0]["ref"], "11.steps")
        self.assertEqual(detected["seed"][0]["maps"][0]["ref"], "11.seed")
        template = payload["suggested_template"]
        self.assertEqual(template["kind"], "txt2img")
        self.assertEqual(template["parameters"]["size"]["maps"][0]["target"]["ref"], "10.width")
        self.assertEqual(template["parameters"]["size"]["default"], "512x512")
        self.assertEqual(template["parameters"]["steps"]["default"], 20)

    def test_workflow_parameters_template_endpoint_returns_copyable_template(self) -> None:
        response = self.client.get(
            f"/v1/admin/workflows/{self.workflow_name}/parameters/template",
            headers={"Authorization": "Bearer admin-token"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["parameter_error"])
        template = payload["template"]
        self.assertEqual(template["version"], 1)
        self.assertEqual(template["kind"], "txt2img")
        self.assertEqual(template["parameters"]["cfg"]["maps"][0]["target"]["ref"], "11.cfg")
        self.assertEqual(template["parameters"]["seed"]["maps"][0]["target"]["ref"], "11.seed")

    def test_images_generations_passes_standard_params_to_job_manager(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-img"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/generations",
                headers={
                    "Authorization": "Bearer secret-token",
                    "x-comfyui-async": "1",
                },
                json={"prompt": "cat", "model": "test_txt2img", "size": "1024x768", "seed": 7, "steps": 12},
            )
        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"], {"size": "1024x768", "seed": 7, "steps": 12})

    def test_images_generations_returns_b64_json_for_base64_response_format(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        done = asyncio.Event()
        done.set()
        job_id = "job-image-b64"
        out_dir = Path(os.environ["RUNS_DIR"]) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "image.png"
        raw_bytes = b"fake-image-bytes"
        out_path.write_bytes(raw_bytes)

        job = Job(
            job_id=job_id,
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2img",
            workflow=self.workflow_name,
            requested_model="test_txt2img",
            outputs=[
                JobOutput(
                    filename="image.png",
                    url=f"/runs/{job_id}/image.png",
                    media_type="image/png",
                    node_id="2",
                    output_key="images",
                )
            ],
            done=done,
        )

        with (
            patch.object(self.app.state.jobs, "create_job", AsyncMock(return_value=SimpleNamespace(job_id=job_id))),
            patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)),
        ):
            response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret-token"},
                json={"prompt": "cat", "model": "test_txt2img", "response_format": "base64"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(base64.b64decode(payload["data"][0]["b64_json"]), raw_bytes)

    def test_images_generations_returns_all_outputs_for_url_and_b64(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        done = asyncio.Event()
        done.set()
        job_id = "job-image-multi"
        out_dir = Path(os.environ["RUNS_DIR"]) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        first = b"first-image-bytes"
        second = b"second-image-bytes"
        (out_dir / "image-1.png").write_bytes(first)
        (out_dir / "image-2.png").write_bytes(second)

        job = Job(
            job_id=job_id,
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2img",
            workflow=self.workflow_name,
            requested_model="test_txt2img",
            url=f"/runs/{job_id}/image-1.png",
            outputs=[
                JobOutput(
                    filename="image-1.png",
                    url=f"/runs/{job_id}/image-1.png",
                    media_type="image/png",
                    node_id="2",
                    output_key="images",
                ),
                JobOutput(
                    filename="image-2.png",
                    url=f"/runs/{job_id}/image-2.png",
                    media_type="image/png",
                    node_id="2",
                    output_key="images",
                ),
            ],
            done=done,
        )

        with (
            patch.object(self.app.state.jobs, "create_job", AsyncMock(return_value=SimpleNamespace(job_id=job_id))),
            patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)),
        ):
            url_response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret-token"},
                json={"prompt": "cat", "model": "test_txt2img", "n": 2},
            )
            b64_response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret-token"},
                json={"prompt": "cat", "model": "test_txt2img", "n": 2, "response_format": "b64_json"},
            )

        self.assertEqual(url_response.status_code, 200)
        url_payload = url_response.json()
        self.assertEqual(len(url_payload["data"]), 2)
        self.assertTrue(all("url" in item for item in url_payload["data"]))
        self.assertIn("/runs/job-image-multi/image-1.png", url_payload["data"][0]["url"])
        self.assertIn("/runs/job-image-multi/image-2.png", url_payload["data"][1]["url"])

        self.assertEqual(b64_response.status_code, 200)
        b64_payload = b64_response.json()
        self.assertEqual(len(b64_payload["data"]), 2)
        self.assertEqual(base64.b64decode(b64_payload["data"][0]["b64_json"]), first)
        self.assertEqual(base64.b64decode(b64_payload["data"][1]["b64_json"]), second)

    def test_images_edits_rejects_missing_prompt(self) -> None:
        mock_create_job = AsyncMock()
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"model": "test_img2img"},
                files={"image": ("one.png", b"only-image", "image/png")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["message"], "Missing 'prompt'")
        mock_create_job.assert_not_awaited()

    def test_images_edits_keeps_dash_prompt_on_job(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-dash"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"prompt": " - ", "model": "test_img2img", "negative_prompt": "-"},
                files={"image": ("one.png", b"only-image", "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["prompt"], "-")
        self.assertEqual(kwargs["negative_prompt"], "-")

    def test_images_edits_rejects_txt2img_workflow_with_clear_400(self) -> None:
        mock_create_job = AsyncMock()
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token"},
                data={"prompt": "cat", "model": "test_txt2img"},
                files={"image": ("input.png", b"fake-image", "image/png")},
            )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("does not support img2img", payload["error"]["message"])
        self.assertIn("missing LoadImage", payload["error"]["message"])
        mock_create_job.assert_not_awaited()

    def test_images_edits_collects_repeated_image_fields_and_sidecar_params(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-multi"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"prompt": "make it cinematic", "model": "test_img2img", "n": "2", "steps": "6"},
                files=[
                    ("image[]", ("one.png", b"first-image", "image/png")),
                    ("image[]", ("two.png", b"second-image", "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["kind"], "img2img")
        self.assertEqual(kwargs["workflow"], self.img2img_workflow_name)
        self.assertEqual(kwargs["prompt"], "make it cinematic")
        self.assertTrue(kwargs["image"])
        self.assertEqual(kwargs["standard_params"]["n"], "2")
        self.assertEqual(kwargs["standard_params"]["steps"], "6")
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 2)
        self.assertTrue(kwargs["standard_params"]["image2"])
        self.assertNotEqual(kwargs["image"], kwargs["standard_params"]["image2"])

    def test_images_edits_accepts_json_images_array(self) -> None:
        first = "data:image/png;base64," + base64.b64encode(b"json-one").decode("ascii")
        second = "data:image/png;base64," + base64.b64encode(b"json-two").decode("ascii")
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-json"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                json={
                    "prompt": "edit both",
                    "model": "test_img2img",
                    "images": [first, second],
                    "seed": 9,
                },
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 2)
        self.assertEqual(kwargs["standard_params"]["seed"], 9)
        self.assertTrue(kwargs["standard_params"]["image2"])

    def test_images_edits_single_image_sets_input_count_default(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-one"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"prompt": "edit one", "model": "test_img2img"},
                files={"image": ("one.png", b"only-image", "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 1)
        self.assertNotIn("image2", kwargs["standard_params"])

    def test_images_edits_collects_indexed_image_fields(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-indexed"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"prompt": "edit indexed", "model": "test_img2img"},
                files=[
                    ("image[0]", ("one.png", b"first-indexed", "image/png")),
                    ("image[1]", ("two.png", b"second-indexed", "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 2)
        self.assertTrue(kwargs["standard_params"]["image2"])
        self.assertNotEqual(kwargs["image"], kwargs["standard_params"]["image2"])

    def test_images_edits_collects_image_plus_image2_fields(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-image2"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"prompt": "edit numbered", "model": "test_img2img"},
                files=[
                    ("image", ("one.png", b"first-numbered", "image/png")),
                    ("image2", ("two.png", b"second-numbered", "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 2)
        self.assertTrue(kwargs["standard_params"]["image2"])
        self.assertNotEqual(kwargs["image"], kwargs["standard_params"]["image2"])

    def test_collect_edit_image_items_accepts_client_field_aliases(self) -> None:
        collect = self.app_module.collect_edit_image_items
        self.assertEqual(
            collect(values={"image[0]": "one", "image[1]": "two"}, is_form=True),
            ["one", "two"],
        )
        self.assertEqual(
            collect(values={"image": "one", "image2": "two"}, is_form=True),
            ["one", "two"],
        )
        self.assertEqual(
            collect(values={"image": ["one", "two"]}, is_form=True),
            ["one", "two"],
        )
        self.assertEqual(
            collect(values={"image": "one", "image[]": "two", "image2": "two"}, is_form=True),
            ["one", "two"],
        )
        self.assertEqual(
            collect(values={"input_reference": "one", "image2": "two"}, is_form=True),
            ["one", "two"],
        )
        self.assertEqual(
            collect(values={"input_reference": "one", "input_reference2": "two"}, is_form=True),
            ["one", "two"],
        )
        self.assertEqual(
            collect(values={"input_references": ["one", "two", "three"]}, is_form=False),
            ["one", "two", "three"],
        )

    def test_images_edits_collects_repeated_image_field(self) -> None:
        mock_create_job = AsyncMock(return_value=SimpleNamespace(job_id="job-edits-repeat"))
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer secret-token", "x-comfyui-async": "1"},
                data={"prompt": "edit repeated", "model": "test_img2img"},
                files=[
                    ("image", ("one.png", b"first-repeat", "image/png")),
                    ("image", ("two.png", b"second-repeat", "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 2)
        self.assertTrue(kwargs["standard_params"]["image2"])
        self.assertNotEqual(kwargs["image"], kwargs["standard_params"]["image2"])

    def test_videos_create_passes_duration_and_fps_standard_params(self) -> None:
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(job_id="job-video", requested_model=self.workflow_name, created_at=123)
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/videos",
                headers={"Authorization": "Bearer secret-token"},
                data={
                    "prompt": "cat animation",
                    "model": "test_txt2video",
                    "seconds": "5",
                    "size": "1280x720",
                    "fps": "24",
                    "frames": "120",
                },
                files={},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "queued")
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["standard_params"], {"duration": "5", "size": "1280x720", "fps": "24", "frames": "120"})

    def test_videos_create_accepts_json_body_with_duration_and_size_fields(self) -> None:
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(job_id="job-video-json", requested_model="test_txt2video", created_at=123)
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/videos",
                headers={"Authorization": "Bearer secret-token"},
                json={
                    "prompt": "astronaut walking on the moon",
                    "model": "test_txt2video",
                    "duration": 5,
                    "width": 1280,
                    "height": 720,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "video")
        self.assertEqual(payload["model"], "test_txt2video")
        self.assertEqual(payload["status"], "queued")
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["kind"], "txt2video")
        self.assertEqual(kwargs["workflow"], self.txt2video_workflow_name)
        self.assertEqual(kwargs["requested_model"], "test_txt2video")
        self.assertEqual(kwargs["prompt"], "astronaut walking on the moon")
        self.assertEqual(kwargs["seconds"], "5")
        self.assertEqual(kwargs["standard_params"], {"duration": "5", "width": "1280", "height": "720"})

    def test_videos_create_accepts_hybrid_video_workflow_for_txt2video(self) -> None:
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(job_id="job-hybrid", requested_model=self.hybrid_video_workflow_name, created_at=123)
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/videos",
                headers={"Authorization": "Bearer secret-token"},
                data={
                    "prompt": "cat animation",
                    "model": "test_hybrid_video",
                    "seconds": "5",
                },
                files={},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "queued")
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["workflow"], self.hybrid_video_workflow_name)

    def test_videos_create_collects_custom_secondary_prompt_and_image_parameters(self) -> None:
        first_image = "data:image/png;base64," + base64.b64encode(b"first-image").decode("ascii")
        second_image = "data:image/png;base64," + base64.b64encode(b"second-image").decode("ascii")
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(
                job_id="job-dual-input",
                requested_model=self.dual_input_video_workflow_name,
                created_at=123,
            )
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/videos",
                headers={"Authorization": "Bearer secret-token"},
                json={
                    "prompt": "primary prompt",
                    "prompt2": "secondary prompt",
                    "model": "test_dual_input_video",
                    "image": first_image,
                    "image2": second_image,
                    "duration": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["workflow"], self.dual_input_video_workflow_name)
        self.assertEqual(kwargs["prompt"], "primary prompt")
        self.assertEqual(kwargs["standard_params"]["duration"], "5")
        self.assertEqual(kwargs["standard_params"]["prompt2"], "secondary prompt")
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 2)
        self.assertTrue(kwargs["standard_params"]["image2"])
        self.assertTrue(kwargs["image"])
        self.assertNotEqual(kwargs["image"], kwargs["standard_params"]["image2"])

    def test_videos_create_collects_three_reference_images_and_count(self) -> None:
        first = "data:image/png;base64," + base64.b64encode(b"frame-one").decode("ascii")
        second = "data:image/png;base64," + base64.b64encode(b"frame-two").decode("ascii")
        third = "data:image/png;base64," + base64.b64encode(b"frame-three").decode("ascii")
        mock_create_job = AsyncMock(
            return_value=SimpleNamespace(
                job_id="job-video-three",
                requested_model=self.dual_input_video_workflow_name,
                created_at=123,
            )
        )
        with patch.object(self.app.state.jobs, "create_job", mock_create_job):
            response = self.client.post(
                "/v1/videos",
                headers={"Authorization": "Bearer secret-token"},
                json={
                    "prompt": "multi frame",
                    "model": "test_dual_input_video",
                    "input_reference": first,
                    "image2": second,
                    "image3": third,
                    "seconds": "4",
                    "size": "1280x720",
                },
            )

        self.assertEqual(response.status_code, 200)
        kwargs = mock_create_job.await_args.kwargs
        self.assertEqual(kwargs["kind"], "img2video")
        self.assertEqual(kwargs["standard_params"]["inputImgCount"], 3)
        self.assertEqual(kwargs["standard_params"]["duration"], "4")
        self.assertEqual(kwargs["standard_params"]["size"], "1280x720")
        self.assertTrue(kwargs["standard_params"]["image2"])
        self.assertTrue(kwargs["standard_params"]["image3"])

    def test_videos_get_returns_signed_content_url(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        done = asyncio.Event()
        done.set()
        job = Job(
            job_id="job-video-content",
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2video",
            workflow=self.txt2video_workflow_name,
            requested_model=self.txt2video_workflow_name,
            outputs=[
                JobOutput(
                    filename="clip.mp4",
                    url="/runs/job-video-content/clip.mp4",
                    media_type="video/mp4",
                    node_id="2",
                    output_key="images",
                )
            ],
            done=done,
        )

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            response = self.client.get(
                "/v1/videos/video_job-video-content",
                headers={"Authorization": "Bearer secret-token"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        parsed = urlparse(payload["url"])
        params = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/v1/videos/video_job-video-content/content")
        self.assertIn("exp", params)
        self.assertIn("sig", params)

    def test_videos_get_maps_running_status_to_in_progress(self) -> None:
        from comfyui2api.jobs import Job

        job = Job(
            job_id="job-video-running",
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="running",
            kind="txt2video",
            workflow=self.txt2video_workflow_name,
            requested_model=self.txt2video_workflow_name,
        )

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            response = self.client.get(
                "/v1/videos/video_job-video-running",
                headers={"Authorization": "Bearer secret-token"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "in_progress")
        self.assertEqual(payload["progress"], 0)
        self.assertIsNone(payload["url"])

    def test_videos_content_accepts_query_api_key(self) -> None:
        from comfyui2api.jobs import Job, JobOutput

        done = asyncio.Event()
        done.set()
        job_id = "job-video-download"
        out_dir = Path(os.environ["RUNS_DIR"]) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "clip.mp4"
        out_path.write_bytes(b"fake-video")

        job = Job(
            job_id=job_id,
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="completed",
            kind="txt2video",
            workflow=self.txt2video_workflow_name,
            requested_model=self.txt2video_workflow_name,
            outputs=[
                JobOutput(
                    filename="clip.mp4",
                    url=f"/runs/{job_id}/clip.mp4",
                    media_type="video/mp4",
                    node_id="2",
                    output_key="images",
                )
            ],
            done=done,
        )

        with patch.object(self.app.state.jobs, "get_job", AsyncMock(return_value=job)):
            response = self.client.get(f"/v1/videos/video_{job_id}/content?api_key=secret-token")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-video")
        self.assertEqual(response.headers["content-type"], "video/mp4")

    def test_websocket_rejects_missing_auth(self) -> None:
        # The socket is accepted first, then closed with a structured JSON
        # error frame (code 1008): closing an unaccepted socket used to make
        # uvicorn answer the handshake with HTTP 403, so browsers only saw a
        # generic transport error.
        with self.client.websocket_connect("/v1/jobs/missing-job/ws") as ws:
            payload = ws.receive_json()
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["data"]["status"], 401)
        self.assertEqual(ctx.exception.code, 1008)

    def test_websocket_accepts_query_token(self) -> None:
        with self.client.websocket_connect("/v1/jobs/missing-job/ws?api_key=secret-token") as ws:
            payload = ws.receive_json()
        self.assertEqual(payload["type"], "error")
        self.assertIn("Job not found", payload["data"]["message"])

    def test_images_generations_failure_includes_job_id(self) -> None:
        from comfyui2api.jobs import Job

        done = asyncio.Event()
        done.set()
        failed_job = Job(
            job_id="job-failed",
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="failed",
            kind="txt2img",
            workflow=self.workflow_name,
            error="RuntimeError: prompt resolution failed",
            done=done,
        )

        mock_create_job = AsyncMock(return_value=failed_job)
        mock_get_job = AsyncMock(side_effect=[failed_job, failed_job])
        with patch.object(self.app.state.jobs, "create_job", mock_create_job), patch.object(
            self.app.state.jobs, "get_job", mock_get_job
        ):
            response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret-token"},
                json={"prompt": "cat", "model": "test_txt2img"},
            )

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(payload["error"]["job_id"], "job-failed")
        self.assertIn("RuntimeError: prompt resolution failed", payload["error"]["message"])

    def test_images_generations_upstream_failure_includes_instance_slug(self) -> None:
        from comfyui2api.jobs import Job

        done = asyncio.Event()
        done.set()
        failed_job = Job(
            job_id="job-upstream",
            created_at_utc="2026-03-16T00:00:00Z",
            created_at=123,
            status="failed",
            kind="txt2img",
            workflow=self.workflow_name,
            instance_slug="gpu-a",
            error=(
                "ComfyApiError: ComfyUI /prompt failed: status=502, "
                "url=http://127.0.0.1:8188/prompt, headers={'server': 'nginx'}, body=''"
            ),
            done=done,
        )

        mock_create_job = AsyncMock(return_value=failed_job)
        mock_get_job = AsyncMock(side_effect=[failed_job, failed_job])
        with patch.object(self.app.state.jobs, "create_job", mock_create_job), patch.object(
            self.app.state.jobs, "get_job", mock_get_job
        ):
            response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer secret-token"},
                json={"prompt": "cat", "model": "test_txt2img"},
            )

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "server_error")
        self.assertEqual(payload["error"]["job_id"], "job-upstream")
        self.assertEqual(payload["error"]["upstream"], "comfyui")
        self.assertEqual(payload["error"]["instance_slug"], "gpu-a")
        self.assertIn("ComfyApiError: ComfyUI /prompt failed", payload["error"]["message"])

    def test_public_workflows_are_removed(self) -> None:
        response = self.client.get("/v1/workflows", headers={"Authorization": "Bearer secret-token"})
        self.assertEqual(response.status_code, 404)


class JobManagerErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_job_uses_workflow_default_prompt_and_image_nodes(self) -> None:
        import comfyui2api.jobs as jobs_module
        from comfyui2api.workflow_params import WorkflowParameterSpec

        from tests.helpers import fake_job_deps

        client = SimpleNamespace(
            object_info=AsyncMock(return_value={}),
            queue_prompt=AsyncMock(side_effect=RuntimeError("stop")),
        )
        deps = fake_job_deps(client=client, runs_dir=Path("."))
        manager = jobs_module.JobManager(
            cfg=deps.cfg,
            registry=SimpleNamespace(),
            pool=deps.pool,
            backend=deps.backend,
        )

        spec = WorkflowParameterSpec(
            version=1,
            kind="img2video",
            parameters={},
            path=Path("hybrid.params.json"),
            prompt_node="339.custom_prompt",
            image_node="167.image",
        )
        workflow = SimpleNamespace(
            name="hybrid.json",
            parameter_spec=spec,
            clone_obj=lambda: {"prompt": {"167": {"class_type": "LoadImage", "inputs": {"image": "input.png"}}}},
        )

        job = await manager.create_job(
            kind="img2video",
            workflow="hybrid.json",
            prompt="hello",
            image="comfyui2api/input.png",
        )

        with patch.object(manager, "_resolve_workflow", AsyncMock(return_value=workflow)), patch.object(
            jobs_module, "prepare_prompt", return_value=({}, None, [], {})
        ) as mock_prepare:
            with self.assertRaises(RuntimeError) as ctx:
                await manager._run_job(job.job_id, client=client)

        self.assertEqual(str(ctx.exception), "stop")
        kwargs = mock_prepare.call_args.kwargs
        self.assertEqual(kwargs["positive_prompt_node"], "339.custom_prompt")
        self.assertEqual(kwargs["image_node"], "167.image")
        self.assertIsNone(kwargs["negative_prompt_node"])

    async def test_fail_job_records_error_and_sets_done(self) -> None:
        import comfyui2api.jobs as jobs_module
        from tests.helpers import fake_job_deps

        deps = fake_job_deps()
        manager = jobs_module.JobManager(
            cfg=deps.cfg,
            registry=SimpleNamespace(),
            pool=deps.pool,
            backend=deps.backend,
        )
        job = await manager.create_job(
            kind="txt2img",
            workflow="broken.json",
            requested_model="broken-model",
            prompt="cat",
        )
        await manager.fail_job(job.job_id, "RuntimeError: boom")
        updated = await manager.get_job(job.job_id)
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.status, "failed")
        self.assertEqual(updated.error, "RuntimeError: boom")
        self.assertTrue(updated.done.is_set())


if __name__ == "__main__":
    unittest.main()
