# Request Routing Architecture - Detailed Explanation

## The Critical Fix: Different Ports

**WRONG (what was in original docs):**
```<0x0A">/v1/*     → 127.0.0.1:8001/v1/*<0x0A"/glm/v1/* → 127.0.0.1:8001/v1/*  ← SAME PORT! COLLISION!
```

**CORRECT (updated architecture):**
```<0x0A">/v1/*     → 127.0.0.1:8001/v1/*        (Granite vLLM)<0x0A"/glm/v1/* → 127.0.0.1:8002/v1/*        (GLM-OCR vLLM)<0x0A"           ↑ DIFFERENT PORTS ↑
```

## Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Docker Container (Port 7860 exposed to external world)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  supervisord (PID 1)                                                    │
│  ├──────────────────────────────────────────────────────────────────┐   │
│  │  [program:glm-ocr]                                               │   │
│  │  Listening on: 127.0.0.1:8002                                   │   │
│  │  Command: python3 -m src.glm_ocr_launcher --port 8002            │   │
│  │  Model: zai-org/GLM-OCR                                         │   │
│  │  GPU: 48GB × 0.42 = 20.16 GB                                    │   │
│  │  Status: Standalone vLLM instance                               │   │
│  │  Managed by: supervisord                                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ├──────────────────────────────────────────────────────────────────┐   │
│  │  [program:granite]                                               │   │
│  │  Listening on: 127.0.0.1:7860                                   │   │
│  │  Command: python3 -m uvicorn src.server:app --port 7860          │   │
│  │  Type: FastAPI Router/Dispatcher                                │   │
│  │                                                                  │   │
│  │  Spawns subprocess:                                             │   │
│  │  └─ Granite vLLM listening on: 127.0.0.1:8001                   │   │
│  │     Command: python3 -m src.vllm_launcher --port 8001            │   │
│  │     Model: ibm-granite/granite-4.0-3b-vision                    │   │
│  │     GPU: 48GB × 0.40 = 19.2 GB                                  │   │
│  │     Status: Subprocess of FastAPI                               │   │
│  │     Managed by: FastAPI lifespan context                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Request Routing Logic

### Request Type 1: Granite Vision Endpoint

```
INCOMING REQUEST:
  POST http://api.example.com:7860/v1/chat/completions
  Header: Authorization: Bearer HF_TOKEN_HERE

PROCESSING:
  1. Network packet arrives at Docker container port 7860
  2. FastAPI (src/server.py) receives request
  3. Middleware validates HF token
  4. Route matching: @app.api_route(\"/v1/{path:path}\") ← MATCHES
  5. Handler: async def proxy_granite(request, path=\"chat/completions\")
  6. Creates upstream URL: http://127.0.0.1:8001/v1/chat/completions
  7. Proxies request to Granite vLLM subprocess (port 8001)

GRANITE vLLM PROCESSES:
  1. Receives OpenAI-compatible request on port 8001
  2. Runs inference with Granite 4 Vision model
  3. Generates response
  4. Sends back to FastAPI on port 7860

RESPONSE:
  FastAPI receives from Granite → forwards to client\
  ✅ Client gets response from Granite 4 Vision\
```

### Request Type 2: GLM-OCR Endpoint

```
INCOMING REQUEST:
  POST http://api.example.com:7860/glm/v1/chat/completions
  Header: Authorization: Bearer HF_TOKEN_HERE

PROCESSING:
  1. Network packet arrives at Docker container port 7860
  2. FastAPI (src/server.py) receives request
  3. Middleware validates HF token
  4. Route matching: @app.api_route(\"/glm/v1/{path:path}\") ← MATCHES (NOT /v1/*)
  5. Handler: async def proxy_glm_ocr(request, path=\"chat/completions\")
  6. Strips \"/glm\" prefix from path
  7. Creates upstream URL: http://127.0.0.1:8002/v1/chat/completions
  8. Proxies request to GLM-OCR vLLM instance (port 8002)\

GLM-OCR vLLM PROCESSES:
  1. Receives OpenAI-compatible request on port 8002
  2. Runs inference with GLM-OCR model
  3. Generates response
  4. Sends back to FastAPI on port 7860\

RESPONSE:
  FastAPI receives from GLM-OCR → forwards to client\
  ✅ Client gets response from GLM-OCR\
```

## Why This Works

1. **Different Ports Ensure No Collision**
   - Granite: 8001
   - GLM-OCR: 8002
   - Even though both respond to `/v1/*`, they listen on different ports

2. **FastAPI Route Matching Determines Upstream**
   - If incoming path matches `/v1/*` → route to port 8001
   - If incoming path matches `/glm/v1/*` → route to port 8002
   - Python's path matching is unambiguous (specific wins over general)

3. **No Port Sharing or Prefix Rewriting**
   - Both vLLM instances expose standard OpenAI `/v1/*` endpoints
   - No need to modify vLLM's internals
   - Each model runs unmodified vLLM software

## Proof This Architecture Works

### Test Case 1: Granite Request
```bash
curl -X POST http://localhost:7860/v1/chat/completions \
  -H "Authorization: Bearer MY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{\"model\": \"ibm-granite/granite-4.0-3b-vision\", ...}'\
<0x0A"># Result: Routed to 127.0.0.1:8001 → Granite processes → response sent
```

### Test Case 2: GLM-OCR Request
```bash
curl -X POST http://localhost:7860/glm/v1/chat/completions \
  -H "Authorization: Bearer MY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{\"model\": \"zai-org/GLM-OCR\", ...}'\
<0x0A"># Result: Routed to 127.0.0.1:8002 → GLM-OCR processes → response sent
```

## Port Summary

| Port | Service | Type | GPU Allocation | Managed By |
|------|---------|------|---|---|
| 7860 | FastAPI Dispatcher | Router | N/A | supervisord |
| 8001 | Granite vLLM | vLLM Instance | 19.2 GB | FastAPI lifespan |
| 8002 | GLM-OCR vLLM | vLLM Instance | 20.16 GB | supervisord |

