# Checkpoint: OCR VLM Aggregation Pipeline – vLLM Serving Implementation

**Created:** 2026-04-30
**Project:** `ocr_vlm_aggr` (Hugging Face Space: `encryptd/ocr_vlm_aggr`)
**Branch:** `main`
**Status:** Implementation complete and pushed to remote. Pending Docker build verification on HF Space.

---

## 1. What Was Implemented

### A. Core vLLM Model Implementation (`granite4_vision.py`)
- Custom vLLM model class `Granite4VisionForConditionalGeneration` for `ibm-granite/granite-4.0-3b-vision`.
- Architecture:
  - SigLIP vision tower via `init_vision_tower_for_llava`.
  - `WindowQFormerDownsampler` with interpolation + spatial-offset modes.
  - Deepstack injection: vision features projected and injected at specific LLM layers during `forward()`.
  - Spatial sampling: 4 offset groups from last vision layer.
  - `GraniteMoeHybrid` backbone with `embedding_multiplier` / `residual_multiplier`.
- LoRA support:
  - Full-merge mode via `--hf-overrides '{"adapter_path": "..."}'` (`_apply_adapter()` + `_merge_lora_deltas()`).
  - Native runtime LoRA via `--enable-lora` (class inherits `SupportsLoRA`).
  - LM-only adapters only; `modules_to_save` is rejected with `ValueError`.
- Weight mapping (`hf_to_vllm_mapper`) handles HF → vLLM prefix translation.
- Tensor-parallel slicing via `_get_shard_offset_mapping()` for packed QKV.

### B. Serving Infrastructure

#### Embedded Mode (Production)
- `src/server.py` – FastAPI app with:
  - HF token auth middleware (`validate_hf_token`).
  - Lifespan manager launching vLLM as subprocess via `src.vllm_launcher`.
  - 15-min startup timeout, streaming proxy to vLLM OpenAI API.
- `src/vllm_launcher.py` – registers `Granite4VisionForConditionalGeneration` with `ModelRegistry`, then delegates to `vllm.entrypoints.openai.api_server`.
- `src/__init__.py` – package marker.

#### Standalone Mode (Debug)
- `start_granite4_vision_server.py` – direct model registration + vLLM API server.

### C. Docker Deployment
- `Dockerfile`:
  - Base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
  - `COPY` of `granite4_vision.py`, `start_granite4_vision_server.py`, `src/`.
  - Port `8000`, healthcheck on `/health`, CMD runs `uvicorn src.server:app`.
- `requirements.txt`: `vllm`, `fastapi`, `uvicorn[standard]`, `httpx`.

### D. Test Suite
- `tests/test_vllm_server.py` – 17 tests covering:
  - File existence checks.
  - Server command construction.
  - LoRA key remapping (`_peft_to_vllm`).
  - Weight mapper prefix rules.
  - Model registration with vLLM `ModelRegistry`.
  - Mixin compliance (`SupportsLoRA`, `SupportsMultiModal`, `SupportsPP`).
  - Placeholder strings (`<image>`).
  - Adapter loader rejection of missing paths.
  - Downsampler shape smoke tests (`InterpolateDownsampler`, `SpatialOffsetDownsampler`).
  - Launcher subprocess registration.
- vLLM-dependent tests auto-skip when unavailable (e.g., macOS without CUDA).

### E. Skill Documentation
- `.agents/skills/serve-granite4-vision-vllm/SKILL.md` – canonical reference for:
  - Architecture overview.
  - Serving modes (embedded / standalone).
  - LoRA configuration (full-merge vs native).
  - Weight loading and mapping.
  - Deepstack forward pass specifics.
  - Docker deployment checklist.
  - Important notes (canonical file, LM-only adapters, TP slicing).

---

## 2. Files Modified / Created

| File | Status | Notes |
|---|---|---|
| `granite4_vision.py` | Created | Core vLLM model implementation |
| `src/server.py` | Created | FastAPI proxy + vLLM subprocess manager |
| `src/vllm_launcher.py` | Created | Model registration + delegation |
| `src/__init__.py` | Created | Package marker |
| `start_granite4_vision_server.py` | Created | Standalone launcher |
| `Dockerfile` | Modified | CUDA runtime, COPY of model files, healthcheck |
| `requirements.txt` | Modified | `vllm`, `fastapi`, `uvicorn[standard]`, `httpx` |
| `tests/test_vllm_server.py` | Created | 17 tests, auto-skip without vLLM |
| `.agents/skills/serve-granite4-vision-vllm/SKILL.md` | Created | Skill documentation |

---

## 3. Verification Status

| Check | Status |
|---|---|
| Skill requirements ↔ Source code cross-check | **PASS** – all 6 sections verified |
| Remote repo contains all files | **PASS** – `git ls-tree origin/main` confirms |
| `Dockerfile` copies `granite4_vision.py` locally | **PASS** – `COPY granite4_vision.py ./` present |
| `src/server.py` constructs correct vLLM command | **PASS** – `--trust-remote-code`, `--hf-overrides`, `--served-model-name` all present |
| Local test suite (file existence + command construction) | **PASS** – 4/4 tests pass on macOS |
| vLLM-dependent tests (model registration, downsamplers, LoRA mapping) | **SKIPPED** on macOS (no CUDA/vLLM); will run in Docker/CI |
| HF Space Docker build | **PENDING** – push succeeded, build status unknown |
| HF Space runtime (model download + compilation + `/health`) | **PENDING** – requires successful build first |

---

## 4. Known Issues / Next Steps

1. **HF Space build verification**: The remote repo is up to date, but the Hugging Face Space may have build/runtime issues. Check Space logs for:
   - vLLM compilation errors.
   - Model download failures (ensure `HF_TOKEN` env var is set in Space settings).
   - CUDA/GPU availability errors.
2. **Local test execution with vLLM**: The 12 skipped vLLM-dependent tests need to run in a CUDA environment (Docker container or GPU runner).
3. **OCR benchmark integration**: The `run-ocr-benchmark` skill exists but has not been integrated with the vLLM serving pipeline yet.
4. **Aggregation strategy extension**: The `add-aggregation-strategy` skill exists for future voting strategy additions.

---

## 5. Quick Reference for New Sessions

### Key File Paths (relative to repo root)
```
granite4_vision.py              → Custom vLLM model
src/server.py                   → FastAPI proxy (embedded mode)
src/vllm_launcher.py            → Model registration launcher
start_granite4_vision_server.py → Standalone launcher
Dockerfile                      → CUDA runtime deployment
tests/test_vllm_server.py       → Verification test suite
.agents/skills/serve-granite4-vision-vllm/SKILL.md  → Canonical skill reference
```

### Critical Constraints (from SKILL.md)
- `granite4_vision.py` is the **canonical** implementation – do NOT replace with upstream vLLM built-in unless verified.
- `forward()` runs the LLM layer loop **directly** – do NOT refactor into a simple wrapper.
- LoRA adapters must be **LM-only** – `modules_to_save` is unsupported and raises `ValueError`.
- Under tensor parallelism, `_get_shard_offset_mapping()` handles packed QKV slicing.

### Environment Variables for HF Space
- `HF_API_TOKEN` – used by `src/server.py` for request auth + forwarded as `HF_TOKEN`/`HUGGINGFACE_TOKEN` for model download.
- `VLM_MODEL_NAME` – defaults to `ibm-granite/granite-4.0-3b-vision`.

### Serving Commands

**Embedded (production):**
```bash
python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```

**Standalone (debug):**
```bash
python start_granite4_vision_server.py \
    --model ibm-granite/granite-4.0-3b-vision \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --hf-overrides '{"adapter_path": "ibm-granite/granite-4.0-3b-vision"}'
```

---

## 6. Remote Status

- **Remote URL**: `https://huggingface.co/spaces/encryptd/ocr_vlm_aggr`
- **Git remote**: `https://huggingface.co/spaces/encryptd/ocr_vlm_aggr`
- **Latest pushed commit**: `ca3b8f8` (includes test suite + all serving logic)
- **HF Space build**: Check logs at `https://huggingface.co/spaces/encryptd/ocr_vlm_aggr/settings/logs`
