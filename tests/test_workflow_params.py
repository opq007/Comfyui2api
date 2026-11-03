from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comfyui2api.workflow_params import (
    WorkflowParameterDefinition,
    WorkflowParamTarget,
    apply_img2img_resize_params,
    detect_parameter_candidates,
    generate_parameter_template,
    load_workflow_parameter_spec,
    normalize_parameter_value,
    resolve_standard_overrides,
)


class WorkflowParameterMappingTests(unittest.TestCase):
    def test_sidecar_mapping_converts_size_fps_duration_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            workflows_dir = root / "workflows"
            sidecar_dir = workflows_dir / ".comfyui2api"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            sidecar_dir.mkdir(parents=True, exist_ok=True)

            workflow_path = workflows_dir / "video_flow.json"
            workflow_obj = {
                "prompt": {
                    "10": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {"width": 512, "height": 512},
                        "_meta": {"title": "Latent Size"},
                    },
                    "11": {
                        "class_type": "KSampler",
                        "inputs": {"seed": 1},
                        "_meta": {"title": "Sampler"},
                    },
                    "20": {
                        "class_type": "VideoCombine",
                        "inputs": {"fps": 12, "frames": 48},
                        "_meta": {"title": "Video Output"},
                    },
                    "30": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "sample"}},
                }
            }
            workflow_path.write_text(json.dumps(workflow_obj, ensure_ascii=False), encoding="utf-8")

            sidecar = {
                "version": 1,
                "kind": "txt2video",
                "parameters": {
                    "size": {
                        "type": "size",
                        "maps": [
                            {"target": "10.width", "part": "width"},
                            {
                                "target": {
                                    "selector": {
                                        "class_type": "EmptyLatentImage",
                                        "title": "Latent Size",
                                        "input_key": "height",
                                    }
                                },
                                "part": "height",
                            },
                        ],
                    },
                    "fps": {
                        "type": "int",
                        "default": 12,
                        "maps": [
                            {
                                "target": {
                                    "selector": {
                                        "class_type": "VideoCombine",
                                        "title": "Video Output",
                                        "input_key": "fps",
                                    }
                                }
                            }
                        ],
                    },
                    "duration": {
                        "type": "float",
                        "maps": [
                            {"target": "20.frames", "transform": "seconds_to_frames", "fps_param": "fps", "round": "ceil"}
                        ],
                    },
                    "seed": {
                        "type": "int",
                        "maps": [{"target": "11.seed"}],
                    },
                },
            }
            (sidecar_dir / "video_flow.params.json").write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")

            spec = load_workflow_parameter_spec(
                workflows_dir=workflows_dir,
                workflow_path=workflow_path,
                expected_kind="txt2video",
            )
            self.assertIsNotNone(spec)

            overrides = resolve_standard_overrides(
                workflow_obj=workflow_obj,
                spec=spec,
                request_params={"size": "1024x768", "fps": 24, "duration": 5, "seed": 7},
            )

            self.assertEqual(
                overrides,
                [
                    ("10", "width", 1024),
                    ("10", "height", 768),
                    ("11", "seed", 7),
                    ("20", "fps", 24),
                    ("20", "frames", 120),
                ],
            )

            detected = detect_parameter_candidates(workflow_obj)
            self.assertEqual(detected["size"][0]["maps"][0]["ref"], "10.width")
            self.assertEqual(detected["size"][0]["maps"][1]["ref"], "10.height")
            self.assertEqual(detected["fps"][0]["maps"][0]["ref"], "20.fps")
            self.assertEqual(detected["duration"][0]["maps"][0]["ref"], "20.frames")
            self.assertEqual(detected["duration"][0]["maps"][0]["transform"], "seconds_to_frames")
            self.assertEqual(detected["duration"][0]["paired_fps_ref"], "20.fps")

            template = generate_parameter_template(workflow_obj=workflow_obj, kind="txt2video", spec=spec)
            self.assertEqual(template["parameters"]["size"]["maps"][0]["target"]["ref"], "10.width")
            self.assertEqual(template["parameters"]["fps"]["default"], 12)
            self.assertEqual(template["parameters"]["duration"]["maps"][0]["transform"], "seconds_to_frames")
            self.assertEqual(template["parameters"]["duration"]["maps"][0]["fps_param"], "fps")

    def test_z_image_turbo_fp16_sidecar_maps_size_n_and_seed(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workflows_dir = project_root / "comfyui-api-workflows"
        workflow_path = workflows_dir / "z_image_turbo_fp16.json"
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))

        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="txt2img",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompt_node, "67.text")

        overrides = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={"size": "768x512", "n": 3, "seed": 42, "steps": 6, "cfg": 1.5},
        )
        self.assertEqual(
            overrides,
            [
                ("68", "width", 768),
                ("68", "height", 512),
                ("68", "batch_size", 3),
                ("70", "steps", 6),
                ("70", "cfg", 1.5),
                ("70", "seed", 42),
            ],
        )

        width_only = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={"width": 640, "height": 480},
        )
        self.assertIn(("68", "width", 640), width_only)
        self.assertIn(("68", "height", 480), width_only)

        detected = detect_parameter_candidates(workflow_obj)
        self.assertEqual(detected["size"][0]["maps"][0]["ref"], "68.width")
        self.assertEqual(detected["n"][0]["maps"][0]["ref"], "68.batch_size")
        self.assertEqual(detected["seed"][0]["maps"][0]["ref"], "70.seed")

    def test_kaggle_flux2_edit_sidecar_maps_second_image_and_input_count(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workflows_dir = project_root / "comfyui-api-workflows"
        workflow_path = workflows_dir / "kaggle_flux2_klein_9b_kv_image_edit_api.json"
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))

        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="img2img",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompt_node, "135.text")
        self.assertEqual(spec.image_node, "76.image")
        self.assertEqual(spec.parameters["inputImgCount"].default, 1)

        single = resolve_standard_overrides(workflow_obj=workflow_obj, spec=spec, request_params={})
        self.assertIn(("705", "value", 1), single)

        overrides = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={
                "n": 2,
                "steps": 8,
                "cfg": 1.2,
                "seed": 11,
                "image2": "uploads/second.png",
                "inputImgCount": 2,
            },
        )
        self.assertEqual(
            overrides,
            [
                ("129", "batch_size", 2),
                ("137", "steps", 8),
                ("138", "cfg", 1.2),
                ("125", "noise_seed", 11),
                ("718", "value", False),
                ("81", "image", "uploads/second.png"),
                ("705", "value", 2),
            ],
        )

        sized = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params=apply_img2img_resize_params({"size": "1024x768"}),
        )
        self.assertIn(("718", "value", True), sized)
        self.assertIn(("720", "value", 1024), sized)
        self.assertIn(("721", "value", 768), sized)

        both_dims = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params=apply_img2img_resize_params({"width": 640, "height": 480}),
        )
        self.assertIn(("718", "value", True), both_dims)
        self.assertIn(("720", "value", 640), both_dims)
        self.assertIn(("721", "value", 480), both_dims)

        width_only = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params=apply_img2img_resize_params({"width": 1024, "resizeFlag": True}),
        )
        self.assertIn(("718", "value", False), width_only)
        self.assertNotIn(("720", "value", 1024), width_only)

    def test_kaggle_minimax_ti2v_sidecar_maps_duration_size_and_input_count(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workflows_dir = project_root / "comfyui-api-workflows"
        workflow_path = workflows_dir / "kaggle_minimax_h3_ti2v_api.json"
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))

        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="img2video",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompt_node, "133.prompt")
        self.assertEqual(spec.image_node, "114.image")
        self.assertEqual(spec.parameters["inputImgCount"].default, 0)

        text_only = resolve_standard_overrides(workflow_obj=workflow_obj, spec=spec, request_params={})
        self.assertIn(("186", "value", 0), text_only)

        overrides = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={
                "duration": 5,
                "size": "1280x720",
                "fps": 24,
                "seed": 9,
                "image2": "uploads/last.png",
                "inputImgCount": 2,
            },
        )
        self.assertIn(("135", "value", 5.0), overrides)
        self.assertIn(("170", "value", 1280), overrides)
        self.assertIn(("171", "value", 720), overrides)
        self.assertIn(("132", "fps", 24.0), overrides)
        self.assertIn(("131", "noise_seed", 9), overrides)
        self.assertIn(("191", "image", "uploads/last.png"), overrides)
        self.assertIn(("186", "value", 2), overrides)

        three_images = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={"inputImgCount": 3, "image2": "uploads/last.png", "image3": "uploads/extra.png"},
        )
        self.assertIn(("186", "value", 3), three_images)
        self.assertIn(("191", "image", "uploads/last.png"), three_images)
        self.assertFalse(any(item[0] == "image3" or item[2] == "uploads/extra.png" for item in three_images))

    def test_qwen_image21_aio_sidecar_maps_constants_images_and_count(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workflows_dir = project_root / "comfyui-api-workflows"
        workflow_path = workflows_dir / "qwen_image21_turbo_aio_api.json"
        sidecar_path = workflows_dir / ".comfyui2api" / f"{workflow_path.stem}.params.json"
        if not workflow_path.exists() or not sidecar_path.exists():
            self.skipTest(f"Workflow fixture not present: {workflow_path.name}")
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))

        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="img2img",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompt_node, "541.prompt")
        self.assertEqual(spec.negative_prompt_node, "541.negative_prompt")
        self.assertEqual(spec.image_node, "470.image")
        # Fixed 8-step ManualSigmas schedule: there is no sampler steps input.
        self.assertNotIn("steps", spec.parameters)

        overrides = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={
                "size": "768x1024",
                "n": 2,
                "cfg": 1.5,
                "seed": 42,
                "image2": "uploads/second.png",
                "image3": "uploads/third.png",
                "inputImgCount": 3,
            },
        )
        self.assertEqual(
            overrides,
            [
                ("535", "value", 768),
                ("536", "value", 1024),
                ("511", "batch_size", 2),
                ("526", "cfg", 1.5),
                ("524", "noise_seed", 42),
                ("544", "image", "uploads/second.png"),
                ("547", "image", "uploads/third.png"),
                ("537", "value", 3),
            ],
        )

        text_only = resolve_standard_overrides(workflow_obj=workflow_obj, spec=spec, request_params={})
        self.assertEqual(text_only, [("537", "value", 0)])

    def test_moss_image_aio_sidecar_routes_negative_and_branch_params(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workflows_dir = project_root / "comfyui-api-workflows"
        workflow_path = workflows_dir / "moss_image_aio_api.json"
        sidecar_path = workflows_dir / ".comfyui2api" / f"{workflow_path.stem}.params.json"
        if not workflow_path.exists() or not sidecar_path.exists():
            self.skipTest(f"Workflow fixture not present: {workflow_path.name}")
        workflow_obj = json.loads(workflow_path.read_text(encoding="utf-8"))

        spec = load_workflow_parameter_spec(
            workflows_dir=workflows_dir,
            workflow_path=workflow_path,
            expected_kind="img2img",
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.prompt_node, "41.value")
        self.assertEqual(spec.negative_prompt_node, "47.text")
        self.assertEqual(spec.negative_prompt_node_with_image, "36.prompt")
        self.assertEqual(spec.image_node, "8.image")
        self.assertEqual(spec.negative_prompt_node_for(image_count=0), "47.text")
        self.assertEqual(spec.negative_prompt_node_for(image_count=1), "36.prompt")

        overrides = resolve_standard_overrides(
            workflow_obj=workflow_obj,
            spec=spec,
            request_params={
                "size": "1344x768",
                "n": 2,
                "steps": 8,
                "cfg": 1.2,
                "seed": 7,
                "inputImgCount": 1,
            },
        )
        self.assertEqual(
            overrides,
            [
                ("56", "value", 1344),
                ("57", "value", 768),
                ("28", "batch_size", 2),
                ("37", "steps", 8),
                ("46", "steps", 8),
                ("37", "cfg", 1.2),
                ("46", "cfg", 1.2),
                ("72", "seed", 7),
                ("52", "value", 1),
            ],
        )

        text_only = resolve_standard_overrides(workflow_obj=workflow_obj, spec=spec, request_params={})
        self.assertEqual(text_only, [("52", "value", 0)])

    def test_detection_infers_constant_node_parameters_and_pairs_size(self) -> None:
        workflow_obj = {
            "prompt": {
                "1": {"class_type": "INTConstant", "inputs": {"value": 720}, "_meta": {"title": "width"}},
                "2": {"class_type": "INTConstant", "inputs": {"value": 1280}, "_meta": {"title": "height"}},
                "3": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": ["1", 0], "height": ["2", 0], "batch_size": 1},
                },
                "4": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 4}},
                "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
                "6": {"class_type": "INTConstant", "inputs": {"value": 4}, "_meta": {"title": "batch_size"}},
            }
        }

        detected = detect_parameter_candidates(workflow_obj)
        self.assertEqual(detected["width"][0]["maps"][0]["ref"], "1.value")
        self.assertEqual(detected["height"][0]["maps"][0]["ref"], "2.value")
        self.assertEqual(detected["size"][0]["maps"][0]["ref"], "1.value")
        self.assertEqual(detected["size"][0]["maps"][1]["ref"], "2.value")
        self.assertEqual(detected["size"][0]["maps"][0]["part"], "width")
        self.assertEqual(detected["size"][0]["maps"][1]["part"], "height")
        self.assertEqual(detected["n"][0]["maps"][0]["ref"], "3.batch_size")
        self.assertIn("6.value", [m["ref"] for entry in detected["n"] for m in entry["maps"]])
        self.assertEqual(detected["steps"][0]["maps"][0]["ref"], "4.steps")

        template = generate_parameter_template(workflow_obj=workflow_obj, kind="txt2img")
        self.assertEqual(template["parameters"]["size"]["default"], "720x1280")
        self.assertEqual(template["parameters"]["width"]["default"], 720)
        self.assertEqual(template["parameters"]["height"]["default"], 1280)

    def test_detection_pairs_width_and_height_across_nodes(self) -> None:
        # Width and height live on different primitive nodes but feed the same
        # latent node; the pair must be detected even though neither node is a
        # titled constant.
        workflow_obj = {
            "prompt": {
                "1": {"class_type": "PrimitiveInt", "inputs": {"width": 512}},
                "2": {"class_type": "PrimitiveInt", "inputs": {"height": 768}},
                "3": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": ["1", 0], "height": ["2", 0], "batch_size": 1},
                },
            }
        }

        detected = detect_parameter_candidates(workflow_obj)
        self.assertEqual(detected["size"][0]["maps"][0]["ref"], "1.width")
        self.assertEqual(detected["size"][0]["maps"][1]["ref"], "2.height")
        self.assertEqual(detected["size"][0]["reason"], "paired width and height inputs feeding the same node")
        self.assertEqual(detected["width"][0]["maps"][0]["ref"], "1.width")
        self.assertEqual(detected["height"][0]["maps"][0]["ref"], "2.height")

    def test_apply_img2img_resize_params_forces_flag_from_complete_size(self) -> None:
        self.assertEqual(apply_img2img_resize_params({}), {"resizeFlag": False})
        self.assertEqual(
            apply_img2img_resize_params({"size": "1024x768", "resizeFlag": False}),
            {"size": "1024x768", "resizeFlag": True},
        )
        self.assertEqual(
            apply_img2img_resize_params({"width": 640, "height": 480}),
            {"width": 640, "height": 480, "resizeFlag": True},
        )
        self.assertEqual(
            apply_img2img_resize_params({"width": 1024, "resizeFlag": True}),
            {"resizeFlag": False},
        )
        self.assertEqual(
            apply_img2img_resize_params({"size": "  ", "width": "", "resizeFlag": True}),
            {"size": "  ", "width": "", "resizeFlag": False},
        )

    def test_bool_parameter_normalizes_common_truthy_values(self) -> None:
        spec = WorkflowParameterDefinition(
            name="resizeFlag",
            type="bool",
            maps=(WorkflowParamTarget(ref="718.value"),),
        )
        self.assertIs(normalize_parameter_value(spec, True), True)
        self.assertIs(normalize_parameter_value(spec, "false"), False)
        self.assertIs(normalize_parameter_value(spec, 1), True)
        self.assertIs(normalize_parameter_value(spec, "0"), False)
        with self.assertRaises(ValueError):
            normalize_parameter_value(spec, "yes")

    def test_sidecar_mapping_loads_explicit_prompt_and_image_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            workflows_dir = root / "workflows"
            sidecar_dir = workflows_dir / ".comfyui2api"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            sidecar_dir.mkdir(parents=True, exist_ok=True)

            workflow_path = workflows_dir / "hybrid_flow.json"
            workflow_obj = {
                "prompt": {
                    "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
                    "2": {"class_type": "CustomPromptNode", "inputs": {"custom_prompt": "hello"}},
                    "3": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "sample"}},
                }
            }
            workflow_path.write_text(json.dumps(workflow_obj, ensure_ascii=False), encoding="utf-8")

            sidecar = {
                "version": 1,
                "kind": "img2video",
                "prompt_node": "2.custom_prompt",
                "image_node": "1.image",
                "parameters": {
                    "duration": {
                        "type": "float",
                        "maps": [{"target": "4.value"}],
                    }
                },
            }
            (sidecar_dir / "hybrid_flow.params.json").write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")

            spec = load_workflow_parameter_spec(
                workflows_dir=workflows_dir,
                workflow_path=workflow_path,
                expected_kind="img2video",
            )

            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertEqual(spec.prompt_node, "2.custom_prompt")
            self.assertEqual(spec.image_node, "1.image")
            self.assertEqual(spec.negative_prompt_node, "")

    def test_sidecar_mapping_supports_custom_string_and_image_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            workflows_dir = root / "workflows"
            sidecar_dir = workflows_dir / ".comfyui2api"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            sidecar_dir.mkdir(parents=True, exist_ok=True)

            workflow_path = workflows_dir / "dual_input.json"
            workflow_obj = {
                "prompt": {
                    "437": {"class_type": "LoadImage", "inputs": {"image": "primary.png"}},
                    "440": {"class_type": "LoadImage", "inputs": {"image": "secondary.png"}},
                    "438": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "prompt one"}},
                    "325": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "prompt two"}},
                    "500": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "sample"}},
                }
            }
            workflow_path.write_text(json.dumps(workflow_obj, ensure_ascii=False), encoding="utf-8")

            sidecar = {
                "version": 1,
                "kind": "img2video",
                "prompt_node": "438.value",
                "image_node": "437.image",
                "parameters": {
                    "prompt2": {"type": "string", "maps": [{"target": "325.value"}]},
                    "image2": {"type": "image", "maps": [{"target": "440.image"}]},
                },
            }
            (sidecar_dir / "dual_input.params.json").write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")

            spec = load_workflow_parameter_spec(
                workflows_dir=workflows_dir,
                workflow_path=workflow_path,
                expected_kind="img2video",
            )

            self.assertIsNotNone(spec)
            overrides = resolve_standard_overrides(
                workflow_obj=workflow_obj,
                spec=spec,
                request_params={"prompt2": "secondary prompt", "image2": "uploads/second.png"},
            )

            self.assertEqual(
                overrides,
                [
                    ("325", "value", "secondary prompt"),
                    ("440", "image", "uploads/second.png"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
