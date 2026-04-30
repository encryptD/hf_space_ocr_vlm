---
name: add-vlm-backend
description: Add a new VLM (Vision Language Model) backend to the OCR aggregation pipeline with standard configuration and registration
---

# Add VLM Backend

## When to use
Use this skill when adding a new VLM model/provider to the OCR aggregation pipeline.

## Instructions
1. Ask the user for the backend name, model identifier, and any special configuration (e.g., quantization, context length, API endpoint).
2. Create a new backend module under `src/backends/<backend_name>.py` implementing the standard `BaseBackend` interface.
3. Register the backend in `src/backends/__init__.py` and `src/registry.py`.
4. Add default configuration to `config/models.yaml`.
5. Add a unit test under `tests/backends/test_<backend_name>.py`.
6. Update `README.md` with the new backend in the supported models list.

## Important notes
- Keep backend modules self-contained; all VLM-specific logic goes in the backend file.
- Ensure the backend exposes `generate(image, prompt)` and `health_check()` methods.
- Use existing backends (e.g., `src/backends/qwen_vl.py`) as templates.
- If the model is served via vLLM, use the OpenAI-compatible client path when possible.

## Examples
- "Add Qwen2.5-VL-7B as a new backend"
- "Register a custom vLLM endpoint for OCR"
