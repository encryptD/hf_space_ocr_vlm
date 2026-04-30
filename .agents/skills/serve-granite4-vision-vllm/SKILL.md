---
name: serve-granite4-vision-vllm
description: Serve IBM's granite-4.0-3b-vision VLM model via vLLM using the custom Granite4VisionForConditionalGeneration implementation from granite4_vision.py
---

# Serve Granite 4 Vision via vLLM

## When to use
Use this skill when deploying or serving the IBM `ibm-granite/granite-4.0-3b-vision` model via vLLM in this project. The model requires a custom vLLM implementation because it uses a unique WindowQFormerDownsampler-based deepstack architecture that is not natively supported by upstream vLLM.

## Prerequisites
- The custom model implementation file `granite4_vision.py` must be present in the project root (or copied into the Docker image at `/app`).
- vLLM must be installed with the version compatible with this model's requirements.
- The `start_granite4_vision_server.py` launcher must be available for standalone registration mode.
- The `src/vllm_launcher.py` module launcher must be available for embedded mode (called from `src/server.py`).

## Architecture Overview
The `granite4_vision.py` file implements `Granite4VisionForConditionalGeneration`, a custom vLLM model class with these key components:

- **Vision tower**: SigLIP encoder (initialized via `init_vision_tower_for_llava`)
- **WindowQFormerDownsampler**: Projects vision features into the LLM hidden dimension using a QFormer-based windowed downsampler (supports both interpolation and spatial-offset modes)
- **Deepstack injection**: Vision features from multiple vision layers are projected and injected at specific LLM layers during the forward pass
- **Spatial sampling**: Optional 4-offset spatial sampling groups from the last vision layer injected at additional LLM layers
- **Language model**: GraniteMoeHybrid backbone with `embedding_multiplier` and `residual_multiplier` scaling
- **LoRA support**: Both full-merge at load time and native runtime LoRA serving

## Instructions

### 1. Choose a serving mode

#### Mode A: Embedded vLLM (FastAPI proxy with HF token auth)
This is the default production mode for the Hugging Face Space. It uses `src/server.py` to launch vLLM as a subprocess and proxies requests through FastAPI with token authentication.

- Ensure `src/vllm_launcher.py` is in the `src/` package.
- The launcher registers the custom model with `ModelRegistry.register_model("Granite4VisionForConditionalGeneration", "granite4_vision:Granite4VisionForConditionalGeneration")` before delegating to `vllm.entrypoints.openai.api_server`.
- The `src/server.py` lifespan manager constructs the vLLM command with:
  - `--trust-remote-code`
  - `--model ibm-granite/granite-4.0-3b-vision`
  - `--hf-overrides '{"adapter_path": "ibm-granite/granite-4.0-3b-vision"}'` (for full-merge mode)

#### Mode B: Standalone vLLM server
Use `start_granite4_vision_server.py` directly for development or debugging without the FastAPI proxy.

```bash
python start_granite4_vision_server.py \
    --model ibm-granite/granite-4.0-3b-vision \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --hf-overrides '{"adapter_path": "ibm-granite/granite-4.0-3b-vision"}'
```

### 2. Configure LoRA serving (choose one)

#### Full merge (default, fastest inference)
All LoRA deltas are merged into base weights at load time. Every request uses the merged model.

```bash
--hf-overrides '{"adapter_path": "ibm-granite/granite-4.0-3b-vision"}'
```

The `_apply_adapter()` method in `granite4_vision.py` handles this automatically during `load_weights()`.

#### Native LoRA runtime
vLLM applies the LM LoRA dynamically per request. Text-only prompts use the pure base model; image prompts apply the LoRA adapter at inference time.

```bash
--enable-lora --max-lora-rank 256 \
--default-mm-loras '{"image": "ibm-granite/granite-4.0-3b-vision"}'
```

### 3. Ensure model file availability

The `granite4_vision.py` file must be importable by Python. In the Docker image:
- It should be `COPY`-ed into the container (do NOT download via `huggingface-cli` at build time — the local file is the canonical source).
- It must be on `sys.path` (e.g., at `/app/granite4_vision.py` since `/app` is the working directory).

### 4. Weight loading and mapping

The model uses a custom `WeightsMapper` in `hf_to_vllm_mapper` to translate HF checkpoint keys to vLLM parameter names:
- `model.language_model.` → `language_model.model.`
- `model.layerwise_projectors.` → `layerwise_projectors.`
- `model.spatial_projectors.` → `spatial_projectors.`
- `model.vision_tower.` → `vision_tower.`
- `lm_head.` → `language_model.lm_head.`

LoRA adapter keys are also remapped via `_peft_to_vllm()` to handle the `base_model.model.` prefix and the same HF→vLLM prefix transforms.

### 5. Deepstack forward pass specifics

Unlike standard vLLM models that wrap the entire inner LLM, `Granite4VisionForConditionalGeneration` runs the LLM layer loop **directly** in its `forward()` method. This is necessary because vision features must be injected at specific target layers (from `config.deepstack_layer_map`) during the forward pass, not just at the input embedding stage.

Key implementation details from `granite4_vision.py`:
- `embed_multimodal()` extracts all deepstack features via `_get_all_layer_features()` and stores them in `self._ds_level_features`.
- `embed_input_ids()` converts those into zero-filled buffers in `self._ds_features`, aligned with the vision mask.
- `forward()` iterates `lm_inner.layers[]`, injecting features via `hidden_states[vision_mask] += features[vision_mask]` at each target layer index.

Do NOT refactor this to use a simple wrapper around the inner model — the layer-by-layer injection is architecturally required.

### 6. Docker deployment checklist

- Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (or devel if compiling vLLM)
- `COPY granite4_vision.py ./` into the image
- `COPY start_granite4_vision_server.py ./` into the image
- `COPY src/ ./src/` for the embedded launcher
- Expose port `8000`
- `HEALTHCHECK` on `/health`
- Set `HF_TOKEN` env var for model download at runtime (not build time)

## Important notes
- The `granite4_vision.py` file is the **canonical** vLLM implementation for this model. Do not replace it with an upstream vLLM built-in unless explicitly verified compatible.
- LoRA adapters must be **LM-only** (`modules_to_save` is not supported). The `_apply_adapter()` method will raise a `ValueError` if `modules_to_save` is detected.
- Under tensor parallelism, the LoRA merge logic slices deltas per TP rank using `_get_shard_offset_mapping()` for packed QKV projections.
- The vision tower uses `return_all_hidden_states=True` on the SiglipEncoder to capture intermediate layer outputs for deepstack projection.

## Examples
- "Serve granite-4.0-3b-vision in embedded mode with full-merge LoRA"
- "Start standalone vLLM server for IBM Granite 4 Vision"
- "Deploy the Granite 4 Vision model to the Hugging Face Space"
