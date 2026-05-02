import asyncio
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "ibm-granite/granite-4.0-3b-vision")
_raw_adapter_path = os.environ.get("VLM_ADAPTER_PATH", "").strip()
if _raw_adapter_path:
    VLM_ADAPTER_PATH = _raw_adapter_path
elif "granite-4.0-3b-vision" in VLM_MODEL_NAME:
    # Granite4 Vision requires LM adapter weights for the language backbone.
    # Default to full-merge adapter loading from the same HF repo unless
    # explicitly overridden.
    VLM_ADAPTER_PATH = VLM_MODEL_NAME
else:
    VLM_ADAPTER_PATH = ""
VLLM_PORT = 8001

vllm_process = None
httpx_client = None


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
    global vllm_process, httpx_client

    # Forward auth token for model download
    env = os.environ.copy()
    if HF_API_TOKEN:
        env.setdefault("HF_TOKEN", HF_API_TOKEN)
        env.setdefault("HUGGINGFACE_TOKEN", HF_API_TOKEN)


    # Ensure repo root is on PYTHONPATH so granite4_vision.py is importable
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "src.vllm_launcher",
        "--model", VLM_MODEL_NAME,
        "--port", str(VLLM_PORT),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--enforce-eager",
        "--max-model-len", "8192",
        "--served-model-name", VLM_MODEL_NAME,
    ]
    if VLM_ADAPTER_PATH:
        cmd.extend(["--hf-overrides", f'{{"adapter_path": "{VLM_ADAPTER_PATH}"}}'])
    print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
    vllm_process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    # Wait for vLLM to be ready (allow up to 15 min for model download + compilation)
    httpx_client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{VLLM_PORT}",
        timeout=300.0,
    )
    for _ in range(900):
        try:
            r = await httpx_client.get("/health")
            if r.status_code == 200:
                print("vLLM is ready", flush=True)
                break
        except Exception:
            pass
        await asyncio.sleep(1)
    else:
        print("vLLM did not become ready in time", flush=True)
        raise RuntimeError("vLLM startup timeout")

    yield

    await httpx_client.aclose()
    if vllm_process:
        vllm_process.send_signal(signal.SIGTERM)
        try:
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vllm_process.kill()


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


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    response = await httpx_client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )

    return StreamingResponse(
        response.aiter_raw(),
        status_code=response.status_code,
        headers=dict(response.headers),
    )
