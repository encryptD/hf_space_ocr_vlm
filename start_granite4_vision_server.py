import logging

from vllm import ModelRegistry

logger = logging.getLogger(__name__)

_ARCH = "Granite4VisionForConditionalGeneration"


def _register_if_needed() -> None:
    """Register the custom model only if vLLM doesn't ship one already."""
    try:
        if _ARCH in ModelRegistry.get_supported_archs():
            logger.info("%s is built-in to vLLM – skipping custom registration", _ARCH)
            return
    except Exception:
        pass

    try:
        ModelRegistry.register_model(
            _ARCH,
            "granite4_vision:Granite4VisionForConditionalGeneration",
        )
        logger.info("Registered %s (custom)", _ARCH)
    except Exception as exc:
        logger.warning("Model registration skipped: %s", exc)


def main():
    logging.basicConfig(level=logging.INFO)
    _register_if_needed()

    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
