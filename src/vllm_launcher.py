import logging
import os
import runpy
import sys

from vllm import ModelRegistry

# Dynamically add repo root (parent of this file) so granite4_vision.py is importable
_LAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_LAUNCHER_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def register_model() -> None:
    arch = "Granite4VisionForConditionalGeneration"

    # Always register the repo-local granite4_vision.py implementation.
    # vLLM ≥ 0.20 ships a built-in version but it hard-codes GraniteForCausalLM
    # (dense) as the language backbone. granite-4.0-3b-vision actually uses a
    # GraniteMoeHybrid language model with shared_mlp layers, so the built-in
    # crashes with KeyError: 'layers.0.shared_mlp.input_linear.weight'.
    # Our custom version uses init_vllm_registered_model() which dynamically
    # resolves the correct language model class from text_config.model_type.
    try:
        ModelRegistry.register_model(
            arch,
            "granite4_vision:Granite4VisionForConditionalGeneration",
        )
        logging.info("Registered %s (custom, overriding built-in)", arch)
    except Exception as e:
        logging.warning("Model registration skipped: %s", e)

# Register on import so multiprocessing-spawned engine processes inherit
# the custom architecture mapping without re-running the API server entrypoint.
register_model()


def main() -> None:
    # Delegate to the standard vLLM API server.
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
