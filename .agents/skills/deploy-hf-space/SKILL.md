---
name: deploy-hf-space
description: Package and deploy the OCR VLM aggregation pipeline as a Hugging Face Space with vLLM and token authentication
---

# Deploy to Hugging Face Space

## When to use
Use this skill when preparing the project for deployment as a Hugging Face Space with vLLM backend and HF token authentication.

## Instructions
1. Verify `Dockerfile` exists and uses a CUDA base image compatible with vLLM.
2. Ensure `requirements.txt` includes `vllm`, `transformers`, `Pillow`, and any backend-specific packages.
3. Add or update `app.py` (or the entrypoint script) to:
   - Load all registered backends from `config/models.yaml`.
   - Start the aggregation pipeline on startup.
   - Expose a FastAPI or Gradio interface for OCR requests.
4. Configure HF token authentication via environment variables (`HF_TOKEN`) or `huggingface_hub` login.
5. Create `README.md` Space metadata with the proper YAML frontmatter for Space configuration.
6. Verify `.gitignore` excludes model weights, logs, and local config overrides.

## Important notes
- Do not commit large model files; use `HF_HOME` or model caching in the Dockerfile.
- Ensure the entrypoint handles graceful shutdowns for vLLM engine processes.
- Set `CUDA_VISIBLE_DEVICES` appropriately for the target Space GPU type.

## Examples
- "Deploy the aggregation pipeline to Hugging Face"
- "Set up vLLM in the Space Dockerfile"
