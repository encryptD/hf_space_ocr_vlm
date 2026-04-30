# Convert Streamlit Space to vLLM API
## Problem
Current project is a Streamlit demo app. Need to convert it into an API-only service that serves a VLM OCR model via vLLM, secured with a Hugging Face token.
## Current State
* `src/streamlit_app.py` — Streamlit spiral chart demo
* `Dockerfile` — Python 3.13-slim, exposes port 8501, runs Streamlit
* `requirements.txt` — `altair`, `pandas`, `streamlit`
* `README.md` — HF Space metadata for Streamlit SDK
## Proposed Changes
### 1. Replace `src/streamlit_app.py` with `src/server.py`
* FastAPI app serving vLLM's OpenAI-compatible API
* Add HF token authentication middleware:
    * Reads `HF_API_TOKEN` env var as the expected token
    * Validates `Authorization: Bearer <token>` header on every request
    * Returns 401 if missing or mismatched
    * Skips auth on `/health` endpoint for Docker healthchecks
* On startup, launches vLLM engine with the configured model (env var `VLM_MODEL_NAME`)
* Exposes OpenAI-compatible `/v1/chat/completions` endpoint for OCR inference with image inputs
* Model name placeholder: `MODEL_PLACEHOLDER` (user will provide actual name later)
### 2. Update `Dockerfile`
* Base image: `nvidia/cuda:12.4.1-devel-ubuntu22.04` (needed for vLLM with GPU)
* Install Python 3.11, pip
* Install vLLM, FastAPI, uvicorn
* Expose port 8000
* HEALTHCHECK on `/health`
* ENTRYPOINT runs uvicorn with `src/server.py`
### 3. Update `requirements.txt`
* Replace Streamlit deps with: `vllm`, `fastapi`, `uvicorn[standard]`
### 4. Update `README.md`
* Change `sdk: docker`, `app_port: 8000`
* Remove Streamlit tags
* Update description to reflect OCR VLM API
### 5. Add `.gitignore`
* Ignore `__pycache__`, `*.pyc`, `.env`
## Key Design Decisions
* Model name set via `VLM_MODEL_NAME` env var — easy to swap without code changes
* HF token set via `HF_API_TOKEN` env var (Hugging Face Spaces injects this as a secret)
* Uses vLLM's built-in OpenAI-compatible server pattern via `vllm.entrypoints.openai.api_server` approach, but wrapped in a custom FastAPI for auth middleware
* Port 8000 (standard for FastAPI/uvicorn)
### 6. Serving Modes with vLLM
*Full merge (adapter merged at load time)
*All LoRA deltas are merged into the base weights at load time. This gives the fastest inference, as every request uses the merged model.
##Example
'''
python start_granite4_vision_server.py \
    --model ibm-granite/granite-4.0-3b-vision \
    --trust_remote_code --host 0.0.0.0 --port 8000 \
    --hf-overrides '{"adapter_path": "ibm-granite/granite-4.0-3b-vision"}'
'''

*Native LoRA runtime
*vLLM applies the LM LoRA dynamically per request. Text-only prompts use the pure base model, while image prompts apply the LoRA adapter at inference time.
##Example
'''
python start_granite4_vision_server.py \
    --model ibm-granite/granite-4.0-3b-vision \
    --trust_remote_code --host 0.0.0.0 --port 8000 \
    --enable-lora --max-lora-rank 256 \
    --default-mm-loras '{"image": "ibm-granite/granite-4.0-3b-vision"}'
'''
