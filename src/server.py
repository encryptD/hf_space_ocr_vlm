import asyncio
import base64
import io
import os
import re
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "ibm-granite/granite-4.0-3b-vision")

model = None
processor = None


async def validate_hf_token(request: Request):
    if request.url.path == "/health":
        return
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="HF_API_TOKEN not configured")
    auth = request.headers.get("Authorization", "")
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
    elif len(parts) == 1:
        token = parts[0]
    else:
        token = ""
    if token != HF_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor

    if HF_API_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_API_TOKEN)

    print(f"Loading model {VLM_MODEL_NAME}...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    processor = AutoProcessor.from_pretrained(
        VLM_MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        VLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    print("Model loaded successfully", flush=True)
    yield

    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    try:
        await validate_hf_token(request)
    except HTTPException as exc:
        raise exc
    return await call_next(request)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": VLM_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ibm-granite",
            }
        ],
    }


def decode_image_url(url: str) -> Image.Image:
    """Decode an image URL (base64 data URI or http URL) into a PIL Image."""
    if url.startswith("data:"):
        match = re.match(r"data:image/[^;]+;base64,(.+)", url)
        if not match:
            raise ValueError("Invalid data URI format")
        image_data = base64.b64decode(match.group(1))
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    elif url.startswith(("http://", "https://")):
        data = urllib.request.urlopen(url, timeout=30).read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image URL format: {url[:50]}...")


def prepare_inputs(raw_messages: List[dict]):
    """Convert OpenAI messages to processor-ready conversation and images."""
    conversations = []
    images = []

    for msg in raw_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            conversations.append({"role": role, "content": content})
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                ptype = part.get("type", "")
                if ptype == "text":
                    new_parts.append({"type": "text", "text": part["text"]})
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    img = decode_image_url(url)
                    images.append(img)
                    new_parts.append({"type": "image"})
                else:
                    new_parts.append(part)
            conversations.append({"role": role, "content": new_parts})

    return conversations, images


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.0)
    stream = body.get("stream", False)

    if stream:
        raise HTTPException(status_code=400, detail="Streaming not yet supported")

    try:
        conversations, images = prepare_inputs(messages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid message format: {str(e)}")

    text = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )

    if images:
        inputs = processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
            do_pad=True,
        ).to(model.device)
    else:
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

    def generate():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                use_cache=True,
            )
        return outputs

    outputs = await asyncio.to_thread(generate)

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    response_text = processor.decode(generated, skip_special_tokens=True)

    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = len(generated)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": VLM_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
