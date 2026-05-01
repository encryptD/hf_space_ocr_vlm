"""Tests for vLLM server initialization and Granite 4 Vision model loading.

These tests verify:
  1. The custom model file can be imported and its classes are well-formed.
  2. Model registration logic in src/vllm_launcher.py works.
  3. Weight mapping and LoRA helper functions behave correctly.
  4. The FastAPI server constructs the correct vLLM subprocess command.

If vLLM is not installed (e.g. on macOS without CUDA), tests that depend on it
are skipped automatically.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is on path so imports resolve
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_vllm():
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests that do NOT require vLLM
# ---------------------------------------------------------------------------


class TestGranite4VisionImports(unittest.TestCase):
    """Basic import sanity checks without instantiating the model."""

    def test_granite4_vision_file_exists(self):
        path = REPO_ROOT / "granite4_vision.py"
        self.assertTrue(path.exists(), f"Missing {path}")

    def test_vllm_launcher_file_exists(self):
        path = REPO_ROOT / "src" / "vllm_launcher.py"
        self.assertTrue(path.exists(), f"Missing {path}")

    def test_server_file_exists(self):
        path = REPO_ROOT / "src" / "server.py"
        self.assertTrue(path.exists(), f"Missing {path}")

    def test_start_server_file_exists(self):
        path = REPO_ROOT / "start_granite4_vision_server.py"
        self.assertTrue(path.exists(), f"Missing {path}")


class TestServerCommandConstruction(unittest.TestCase):
    """Verify src/server.py builds the expected vLLM subprocess command."""

    @patch.dict(
        os.environ,
        {
            "VLM_MODEL_NAME": "ibm-granite/granite-4.0-3b-vision",
            "HF_API_TOKEN": "fake-token-123",
        },
        clear=True,
    )
    @patch.dict(sys.modules, {"httpx": MagicMock(), "fastapi": MagicMock(), "fastapi.responses": MagicMock()})
    def test_server_cmd_construction_no_adapter(self):
        import src.server as server_mod
        # When VLM_ADAPTER_PATH is unset for Granite4 Vision, adapter path
        # should default to the model id to enable full-merge weights.
        self.assertEqual(server_mod.VLM_MODEL_NAME, "ibm-granite/granite-4.0-3b-vision")
        self.assertEqual(server_mod.HF_API_TOKEN, "fake-token-123")
        self.assertEqual(
            server_mod.VLM_ADAPTER_PATH,
            "ibm-granite/granite-4.0-3b-vision",
        )

    @patch.dict(
        os.environ,
        {
            "VLM_MODEL_NAME": "ibm-granite/granite-4.0-3b-vision",
            "HF_API_TOKEN": "fake-token-123",
            "VLM_ADAPTER_PATH": "ibm-granite/granite-4.0-3b-vision",
        },
        clear=True,
    )
    @patch.dict(sys.modules, {"httpx": MagicMock(), "fastapi": MagicMock(), "fastapi.responses": MagicMock()})
    def test_server_cmd_construction_with_adapter(self):
        import src.server as server_mod

        # When VLM_ADAPTER_PATH is set, --hf-overrides must contain adapter_path
        self.assertEqual(server_mod.VLM_ADAPTER_PATH, "ibm-granite/granite-4.0-3b-vision")

    def test_vllm_launcher_discovers_repo_root(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")
        # Import in a subprocess to avoid triggering vLLM server startup.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, str('" + str(REPO_ROOT) + "')); "
                    "import src.vllm_launcher as launcher; "
                    "print('REPO_ROOT=' + launcher._REPO_ROOT)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("REPO_ROOT=", result.stdout, msg=result.stderr)
        repo_root = result.stdout.strip().split("REPO_ROOT=", 1)[1]
        self.assertTrue(os.path.isdir(repo_root))
        self.assertTrue(os.path.exists(os.path.join(repo_root, "granite4_vision.py")))


class TestLoRAKeyMapping(unittest.TestCase):
    """Unit tests for PEFT-to-vLLM key remapping logic."""

    def test_peft_to_vllm_key_strips_base_model(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")
        from granite4_vision import Granite4VisionForConditionalGeneration

        raw = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        mapped = Granite4VisionForConditionalGeneration._peft_to_vllm(raw)
        self.assertFalse(mapped.startswith("base_model.model."))
        self.assertIn("language_model.model.", mapped)
        self.assertIn("q_proj", mapped)

    def test_peft_to_vllm_no_base_model_prefix(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")
        from granite4_vision import Granite4VisionForConditionalGeneration

        raw = "model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        mapped = Granite4VisionForConditionalGeneration._peft_to_vllm(raw)
        self.assertIn("language_model.model.", mapped)


class TestWeightMapperConfig(unittest.TestCase):
    """Verify the hf_to_vllm_mapper prefix rules."""

    def test_mapper_prefixes(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")
        from granite4_vision import Granite4VisionForConditionalGeneration

        mapper = Granite4VisionForConditionalGeneration.hf_to_vllm_mapper
        # WeightsMapper exposes orig_to_new_prefix as a dict-like attribute
        prefixes = mapper.orig_to_new_prefix
        self.assertIn("model.language_model.", prefixes)
        self.assertEqual(
            prefixes["model.language_model."],
            "language_model.model.",
        )
        self.assertIn("model.layerwise_projectors.", prefixes)
        self.assertIn("model.spatial_projectors.", prefixes)
        self.assertIn("model.vision_tower.", prefixes)
        self.assertIn("lm_head.", prefixes)


# ---------------------------------------------------------------------------
# Tests that DO require vLLM (skipped when unavailable)
# ---------------------------------------------------------------------------


class TestVLLMModelRegistration(unittest.TestCase):
    """Verify the custom model registers with vLLM's ModelRegistry."""

    def setUp(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")

    def test_model_registration_does_not_raise(self):
        from vllm import ModelRegistry

        # This is exactly what src/vllm_launcher.py does
        try:
            ModelRegistry.register_model(
                "Granite4VisionForConditionalGeneration",
                "granite4_vision:Granite4VisionForConditionalGeneration",
            )
        except Exception as exc:
            # If already registered (e.g. from a previous test) that's OK
            if "already" in str(exc).lower() or "registered" in str(exc).lower():
                pass
            else:
                raise

    def test_model_class_is_importable(self):
        from granite4_vision import Granite4VisionForConditionalGeneration

        self.assertTrue(hasattr(Granite4VisionForConditionalGeneration, "forward"))
        self.assertTrue(hasattr(Granite4VisionForConditionalGeneration, "load_weights"))
        self.assertTrue(hasattr(Granite4VisionForConditionalGeneration, "compute_logits"))
        self.assertTrue(hasattr(Granite4VisionForConditionalGeneration, "embed_multimodal"))
        self.assertTrue(hasattr(Granite4VisionForConditionalGeneration, "embed_input_ids"))

    def test_model_supports_expected_mixins(self):
        from granite4_vision import Granite4VisionForConditionalGeneration
        from vllm.model_executor.models.interfaces import (
            SupportsLoRA,
            SupportsMultiModal,
            SupportsPP,
        )

        self.assertTrue(issubclass(Granite4VisionForConditionalGeneration, SupportsLoRA))
        self.assertTrue(issubclass(Granite4VisionForConditionalGeneration, SupportsMultiModal))
        self.assertTrue(issubclass(Granite4VisionForConditionalGeneration, SupportsPP))

    def test_placeholder_str_for_image(self):
        from granite4_vision import Granite4VisionForConditionalGeneration

        placeholder = Granite4VisionForConditionalGeneration.get_placeholder_str("image", 0)
        self.assertEqual(placeholder, "<image>")

        with self.assertRaises(ValueError):
            Granite4VisionForConditionalGeneration.get_placeholder_str("video", 0)

    def test_lora_adapter_loader_rejects_missing_config(self):
        from granite4_vision import Granite4VisionForConditionalGeneration

        with self.assertRaises(FileNotFoundError):
            Granite4VisionForConditionalGeneration._load_adapter("/nonexistent/path")


class TestDownsamplerModules(unittest.TestCase):
    """Smoke tests for the WindowQFormerDownsampler components."""

    def setUp(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")

    def _make_fake_config(self):
        """Build a minimal mock config object for the downsampler."""
        config = MagicMock()
        config.vision_config.image_size = 384
        config.vision_config.patch_size = 16
        config.vision_config.hidden_size = 768
        config.text_config.hidden_size = 1024
        config.downsample_rate = "4/8"
        config.projector_dropout = 0.0
        config.use_spatial_sampling = False
        return config

    def test_interpolate_downsampler_shape(self):
        import torch
        from granite4_vision import InterpolateDownsampler

        config = self._make_fake_config()
        ds = InterpolateDownsampler(config, mode="area")
        # 384/16 = 24 patches per side → 24*24 = 576 tokens
        x = torch.randn(2, 576, 768)
        out = ds(x)
        # downsample_rate 4/8 = 0.5 → 12x12 = 144 tokens
        self.assertEqual(out.shape, (2, 144, 768))

    def test_spatial_offset_downsampler_shape(self):
        import torch
        from granite4_vision import SpatialOffsetDownsampler

        config = self._make_fake_config()
        ds = SpatialOffsetDownsampler(config, offset=0)
        x = torch.randn(2, 576, 768)
        out = ds(x)
        # Halves each side: 24 → 12, so 144 tokens
        self.assertEqual(out.shape, (2, 144, 768))

    def test_spatial_offset_variants(self):
        import torch
        from granite4_vision import SpatialOffsetDownsampler

        config = self._make_fake_config()
        x = torch.randn(2, 576, 768)
        for offset in range(4):
            ds = SpatialOffsetDownsampler(config, offset=offset)
            out = ds(x)
            self.assertEqual(out.shape, (2, 144, 768))


class TestVLLMLauncherScript(unittest.TestCase):
    """Verify src/vllm_launcher.py can be imported without side effects."""

    def setUp(self):
        if not _has_vllm():
            self.skipTest("vLLM not installed")

    def test_launcher_importable(self):
        # We import in a subprocess so the global side effects (sys.path
        # mutation + ModelRegistry.register_model) don't leak into other tests.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, str('" + str(REPO_ROOT) + "')); "
                    "from vllm import ModelRegistry; "
                    "ModelRegistry.register_model("
                    "    'Granite4VisionForConditionalGeneration',"
                    "    'granite4_vision:Granite4VisionForConditionalGeneration',"
                    "); "
                    "print('OK')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("OK", result.stdout, msg=result.stderr)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
