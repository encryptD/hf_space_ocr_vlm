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


def _is_model_builtin(arch: str) -> bool:
    """Return True if *arch* is already registered in vLLM's model registry."""
    try:
        return arch in ModelRegistry.get_supported_archs()
    except Exception:
        return False


def register_model() -> None:
    arch = "Granite4VisionForConditionalGeneration"

    # vLLM ≥ 0.20 ships with a built-in implementation that accepts the
    # standard (quant_config, cache_config, prefix) constructor signature.
    # Overriding it with the repo-local granite4_vision.py would cause a
    # TypeError because the local version lacks those parameters.
    if _is_model_builtin(arch):
        logging.info(
            "%s is already built-in to vLLM – skipping custom registration", arch
        )
        return

    try:
        ModelRegistry.register_model(
            arch,
            "granite4_vision:Granite4VisionForConditionalGeneration",
        )
        logging.info("Registered %s (custom)", arch)
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
