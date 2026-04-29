import logging
import sys

# Add /app so granite4_vision.py is importable
sys.path.insert(0, "/app")

from vllm import ModelRegistry

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

# Delegate to the standard vLLM API server
import runpy

runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
