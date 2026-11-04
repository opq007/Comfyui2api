from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comfyui2api.comfy_workflow import (
    iter_file_outputs,
    normalize_prompt_enum_inputs,
    prepare_prompt,
    prune_invalid_orphan_output_nodes,
)


class ComfyWorkflowSanitizationTests(unittest.TestCase):
    def test_prepare_prompt_follows_positive_text_source_chain(self) -> None:
        workflow = {
            "prompt": {
                "298": {
                    "class_type": "ComfyMathExpression",
                    "inputs": {"expression": "a", "values.a": ["300", 0]},
                    "_meta": {"title": "Math Expression"},
                },
                "304": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": ["325", 0], "clip": ["278", 0]},
                    "_meta": {"title": "CLIP文本编码"},
                },
                "305": {
                    "class_type": "LTXVConditioning",
                    "inputs": {"frame_rate": ["298", 0], "positive": ["304", 0], "negative": ["315", 0]},
                    "_meta": {"title": "LTXV条件"},
                },
                "316": {
                    "class_type": "CFGGuider",
                    "inputs": {"positive": ["305", 0], "negative": ["305", 1]},
                    "_meta": {"title": "CFG引导器"},
                },
                "315": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "old negative", "clip": ["278", 0]},
                    "_meta": {"title": "CLIP文本编码"},
                },
                "325": {
                    "class_type": "PrimitiveStringMultiline",
                    "inputs": {"value": "old positive"},
                    "_meta": {"title": "Prompt"},
                },
            }
        }

        prompt, extra_data, applied, trace = prepare_prompt(
            workflow_obj=workflow,
            positive_prompt="new positive",
            negative_prompt=None,
            positive_prompt_node=None,
            negative_prompt_node=None,
            image=None,
            image_node=None,
            overrides=[],
        )

        self.assertIsNone(extra_data)
        self.assertEqual(applied, [("325", "value", "new positive")])
        self.assertEqual(prompt["325"]["inputs"]["value"], "new positive")
        self.assertEqual(prompt["315"]["inputs"]["text"], "old negative")
        self.assertEqual(
            trace,
            {
                "positive": [
                    {
                        "node_id": "325",
                        "input_key": "value",
                        "class_type": "PrimitiveStringMultiline",
                        "title": "Prompt",
                        "value": "new positive",
                    }
                ]
            },
        )

    def test_prepare_prompt_routes_negative_to_negative_prompt_key(self) -> None:
        # TextEncodeQwenImage21 carries both prompts on one node, and the
        # CFGGuider feeds outputs 0/1 of it as positive/negative.
        workflow = {
            "prompt": {
                "541": {
                    "class_type": "TextEncodeQwenImage21",
                    "inputs": {"prompt": "old positive", "negative_prompt": "old negative", "clip": ["533", 0]},
                    "_meta": {"title": "文本编码 Qwen Image 2.1"},
                },
                "526": {
                    "class_type": "CFGGuider",
                    "inputs": {"cfg": 1, "positive": ["541", 0], "negative": ["541", 1]},
                    "_meta": {"title": "CFG引导器"},
                },
            }
        }

        prompt, _extra_data, applied, trace = prepare_prompt(
            workflow_obj=workflow,
            positive_prompt="new positive",
            negative_prompt="new negative",
            positive_prompt_node=None,
            negative_prompt_node=None,
            image=None,
            image_node=None,
            overrides=[],
        )

        self.assertEqual(
            applied,
            [("541", "prompt", "new positive"), ("541", "negative_prompt", "new negative")],
        )
        self.assertEqual(prompt["541"]["inputs"]["prompt"], "new positive")
        self.assertEqual(prompt["541"]["inputs"]["negative_prompt"], "new negative")
        self.assertEqual(trace["positive"][0]["input_key"], "prompt")
        self.assertEqual(trace["negative"][0]["input_key"], "negative_prompt")

    def test_prepare_prompt_reports_effective_prompt_targets(self) -> None:
        workflow = {
            "prompt": {
                "1": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "old positive"},
                    "_meta": {"title": "Positive Prompt"},
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "old negative"},
                    "_meta": {"title": "Negative Prompt"},
                },
            }
        }

        prompt, extra_data, applied, trace = prepare_prompt(
            workflow_obj=workflow,
            positive_prompt="new positive",
            negative_prompt="new negative",
            positive_prompt_node=None,
            negative_prompt_node=None,
            image=None,
            image_node=None,
            overrides=[],
        )

        self.assertIsNone(extra_data)
        self.assertEqual(
            applied,
            [("1", "text", "new positive"), ("2", "text", "new negative")],
        )
        self.assertEqual(prompt["1"]["inputs"]["text"], "new positive")
        self.assertEqual(prompt["2"]["inputs"]["text"], "new negative")
        self.assertEqual(
            trace,
            {
                "positive": [
                    {
                        "node_id": "1",
                        "input_key": "text",
                        "class_type": "CLIPTextEncode",
                        "title": "Positive Prompt",
                        "value": "new positive",
                    }
                ],
                "negative": [
                    {
                        "node_id": "2",
                        "input_key": "text",
                        "class_type": "CLIPTextEncode",
                        "title": "Negative Prompt",
                        "value": "new negative",
                    }
                ],
            },
        )

    def test_prepare_prompt_keeps_workflow_defaults_for_dash_sentinel(self) -> None:
        workflow = {
            "prompt": {
                "1": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "preset positive"},
                    "_meta": {"title": "Positive Prompt"},
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "preset negative"},
                    "_meta": {"title": "Negative Prompt"},
                },
            }
        }

        prompt, extra_data, applied, trace = prepare_prompt(
            workflow_obj=workflow,
            positive_prompt=" - ",
            negative_prompt="-",
            positive_prompt_node="1.text",
            negative_prompt_node="2.text",
            image=None,
            image_node=None,
            overrides=[],
        )

        self.assertIsNone(extra_data)
        self.assertEqual(applied, [])
        self.assertEqual(prompt["1"]["inputs"]["text"], "preset positive")
        self.assertEqual(prompt["2"]["inputs"]["text"], "preset negative")
        self.assertEqual(trace, {})

    def test_prepare_prompt_writes_non_sentinel_dash_like_text(self) -> None:
        workflow = {
            "prompt": {
                "1": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "preset positive"},
                    "_meta": {"title": "Positive Prompt"},
                },
                "2": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "preset negative"},
                    "_meta": {"title": "Negative Prompt"},
                },
            }
        }

        prompt, extra_data, applied, trace = prepare_prompt(
            workflow_obj=workflow,
            positive_prompt="---",
            negative_prompt=" - extra",
            positive_prompt_node=None,
            negative_prompt_node=None,
            image=None,
            image_node=None,
            overrides=[],
        )

        self.assertIsNone(extra_data)
        self.assertEqual(applied, [("1", "text", "---"), ("2", "text", " - extra")])
        self.assertEqual(prompt["1"]["inputs"]["text"], "---")
        self.assertEqual(prompt["2"]["inputs"]["text"], " - extra")
        self.assertEqual(trace["positive"][0]["value"], "---")
        self.assertEqual(trace["negative"][0]["value"], " - extra")

    def test_prunes_orphan_output_nodes_missing_required_inputs(self) -> None:
        prompt = {
            "10": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "x",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                },
            },
            "11": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["1", 0],
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "x",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }
        object_info = {
            "VHS_VideoCombine": {
                "output_node": True,
                "input": {
                    "required": {
                        "images": ["IMAGE"],
                        "frame_rate": ["FLOAT", {}],
                        "loop_count": ["INT", {}],
                        "filename_prefix": ["STRING", {}],
                        "format": [["video/h264-mp4"]],
                        "pingpong": ["BOOLEAN", {}],
                        "save_output": ["BOOLEAN", {}],
                    }
                },
            }
        }

        removed = prune_invalid_orphan_output_nodes(prompt, object_info=object_info)

        self.assertEqual(removed, ["10"])
        self.assertNotIn("10", prompt)
        self.assertIn("11", prompt)

    def test_normalizes_enum_inputs_by_basename_when_allowed(self) -> None:
        prompt = {
            "329": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": r"LTX\ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                },
            },
            "134": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "lora_name": r"LTX\ltx-2.3-22b-distilled-lora-384.safetensors",
                    "strength_model": 0.6,
                },
            },
        }
        object_info = {
            "UNETLoader": {
                "input": {
                    "required": {
                        "unet_name": [["ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"]],
                        "weight_dtype": [["default"]],
                    }
                }
            },
            "LoraLoaderModelOnly": {
                "input": {
                    "required": {
                        "lora_name": [["ltx-2.3-22b-distilled-lora-384.safetensors"]],
                        "strength_model": ["FLOAT", {}],
                    }
                }
            },
        }

        changes = normalize_prompt_enum_inputs(prompt, object_info=object_info)

        self.assertEqual(
            changes,
            [
                (
                    "329",
                    "unet_name",
                    r"LTX\ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
                    "ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
                ),
                (
                    "134",
                    "lora_name",
                    r"LTX\ltx-2.3-22b-distilled-lora-384.safetensors",
                    "ltx-2.3-22b-distilled-lora-384.safetensors",
                ),
            ],
        )


class HistoryFileOutputTests(unittest.TestCase):
    """Only files a save node wrote to ComfyUI's output folder are job outputs."""

    def test_keeps_saved_output_and_drops_node_pack_panel_entries(self) -> None:
        # WAS Node Suite republishes its compare-panel previews with
        # type="was_compare" (hash-named .wav, served by its own route), and
        # /view answers 400 "Invalid type" for them — one such entry used to
        # fail the whole TTS job even though the real mp3 was saved.
        history = {
            "outputs": {
                "17": {
                    "audio": [
                        {"filename": "ComfyUI_00001_.mp3", "subfolder": "audio", "type": "output"}
                    ]
                },
                "19": {
                    "audio_sides": [
                        {
                            "filename": "7001648be39185c955473b7fcb6efda8.wav",
                            "subfolder": "",
                            "type": "was_compare",
                        }
                    ]
                },
            }
        }
        self.assertEqual(
            list(iter_file_outputs(history)),
            [
                (
                    "17",
                    "audio",
                    {"filename": "ComfyUI_00001_.mp3", "subfolder": "audio", "type": "output"},
                )
            ],
        )

    def test_normalizes_missing_or_odd_file_info(self) -> None:
        history = {
            "outputs": {
                "1": {
                    "images": [
                        {"filename": "no_type.png", "subfolder": None},
                        {"filename": "preview.png", "subfolder": "sub", "type": "temp"},
                        {"filename": "", "type": "output"},
                        {"filename": 3, "type": "output"},
                        {"subfolder": "", "type": "output"},
                    ],
                    "empty": [],
                    "gifs": [{"filename": "a.gif", "type": "output"}],
                },
                "2": {"images": [{"filename": "b.png", "type": "output"}]},
                "3": "not-a-node",
            }
        }
        self.assertEqual(
            [
                (node_id, key, info["filename"], info["subfolder"], info["type"])
                for node_id, key, info in iter_file_outputs(history)
            ],
            [
                # A missing type is a save node's output and a non-str subfolder
                # is the root; a preview (temp) entry and malformed ones are not
                # job artifacts.
                ("1", "images", "no_type.png", "", "output"),
                ("1", "gifs", "a.gif", "", "output"),
                ("2", "images", "b.png", "", "output"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
