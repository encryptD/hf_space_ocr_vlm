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
    # Try to register the model. If it's already built-in (future vLLM versions),
    # catch the exception gracefully.
    try:
        ModelRegistry.register_model(
            "Granite4VisionForConditionalGeneration",
            "granite4_vision:Granite4VisionForConditionalGeneration",
        )
        logging.info("Registered Granite4VisionForConditionalGeneration")
    except Exception as e:
        logging.warning("Model registration skipped (may already be built-in): %s", e)


def main() -> None:
    register_model()
    # Delegate to the standard vLLM API server.
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
