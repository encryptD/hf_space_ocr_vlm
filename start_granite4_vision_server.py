import logging

from vllm import ModelRegistry

ModelRegistry.register_model(
    "Granite4VisionForConditionalGeneration",
    "granite4_vision:Granite4VisionForConditionalGeneration",
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
