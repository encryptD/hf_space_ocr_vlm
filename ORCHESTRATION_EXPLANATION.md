# Process Orchestration Strategy

## Overview

This document explains the process orchestration strategy for running multiple vLLM instances (Granite 4 Vision + GLM-OCR) in a single Docker container using **supervisord** as the main process manager.

## Core Concept

**supervisord as Docker's PID 1** manages multiple vLLM instances with auto-restart and proper signal handling.

### Before (Current - Single Model)

```
Docker Container
  └─ FastAPI (PID 1, Port 7860)
      └─ Granite vLLM (subprocess)
```

**Problem:** If Granite crashes, there's no auto-restart.

### After (Proposed - Multi-Model)

```
Docker Container
  └─ supervisord (PID 1)
      ├─ [program:glm-ocr] (Priority 998 = start first)
      │  └─ vLLM: zai-org/GLM-OCR @ 127.0.0.1:8002  ← PORT 8002
      │     ├─ GPU: 48 GB × 0.42 = 20.16 GB
      │     ├─ Auto-restart: Yes
      │     └─ startsecs: 60 seconds
      │
      └─ [program:granite] (Priority 999 = start second)
         └─ FastAPI Server @ 127.0.0.1:7860
            ├─ Spawns: Granite vLLM subprocess @ 127.0.0.1:8001  ← PORT 8001
            ├─ GPU: 48 GB × 0.40 = 19.2 GB
            ├─ Routes: /v1/* and /glm/v1/*
            └─ Auto-restart: Yes
```

**Benefits:**
- Any service crashes → auto-restart
- Independent GPU memory allocation
- Separate logging per service
- Graceful shutdown handling

## Startup Timeline

```
T+0s    → Container starts
T+1s    → supervisord becomes PID 1
T+2s    → Spawns GLM-OCR on port 8002 (priority 998)
T+60s   → GLM-OCR ready, spawns FastAPI (priority 999)
T+62s   → FastAPI starts on port 7860, spawns Granite vLLM on port 8001
T+900s  → Granite ready, system fully operational
```

## Port Allocation

**Critical: Each vLLM instance has a DIFFERENT internal port**

```
External (Docker Host)
  └─ :7860 (exposed)
      │
      └─ Container localhost:7860 (FastAPI entry point)
         ├─ Route /v1/* → http://127.0.0.1:8001  (Granite vLLM)
         └─ Route /glm/v1/* → http://127.0.0.1:8002  (GLM-OCR vLLM)
         │
         ├─ Container localhost:8001 (Granite vLLM subprocess)
         │  └─ Managed by FastAPI lifespan
         │
         └─ Container localhost:8002 (GLM-OCR vLLM standalone)
            └─ Managed by supervisord
```

## GPU Memory Allocation

```
Total GPU: 48 GB

Granite: 48 × 0.40 = 19.2 GB (port 8001)
GLM-OCR: 48 × 0.42 = 20.16 GB (port 8002)
Total Used: 39.36 GB (82%)
Headroom: 8.64 GB (17%)
```

## supervisord Configuration

Key parameters in `supervisord.conf`:

```ini
[program:glm-ocr]
command=python3 -m src.glm_ocr_launcher --model zai-org/GLM-OCR --port 8002
priority=998
autostart=true
autorestart=true
startsecs=60

[program:granite]
command=python3 -m uvicorn src.server:app --host 127.0.0.1 --port 7860
priority=999
autostart=true
autorestart=true
```

## Request Routing (Correct)

**For `/v1/*` (Granite):**
```
Client → Port 7860/v1/* → FastAPI → http://127.0.0.1:8001/v1/* → Granite vLLM
```

**For `/glm/v1/*` (GLM-OCR):**
```
Client → Port 7860/glm/v1/* → FastAPI → http://127.0.0.1:8002/v1/* → GLM-OCR vLLM
```

**Key Difference:** FastAPI uses DIFFERENT PORTS to route to different models

### FastAPI Routing Code

```python
# src/server.py

@app.api_route(\"/v1/{path:path}\", methods=[\"GET\", \"POST\", \"PUT\", \"DELETE\"])
async def proxy_granite(request: Request, path: str):
    \"\"\"Route /v1/* to Granite vLLM on port 8001\"\"\"
    response = await httpx_client.request(
        method=request.method,
        url=f\"http://127.0.0.1:8001/v1/{path}\",
        ...
    )
    return response


@app.api_route(\"/glm/v1/{path:path}\", methods=[\"GET\", \"POST\", \"PUT\", \"DELETE\"])
async def proxy_glm_ocr(request: Request, path: str):
    \"\"\"Route /glm/v1/* to GLM-OCR vLLM on port 8002\"\"\"
    response = await httpx_client.request(
        method=request.method,
        url=f\"http://127.0.0.1:8002/v1/{path}\",  ← DIFFERENT PORT
        ...
    )
    return response<0x0A```<0x0A<0x0A## Auto-Recovery Scenarios

| Scenario | Behavior | Impact |
|----------|----------|--------|
| Granite (port 8001) crashes | supervisord respawns | /v1/* recovers in 1-2s |
| GLM-OCR (port 8002) crashes | supervisord respawns | /glm/v1/* recovers in 60s |
| FastAPI (port 7860) crashes | supervisord respawns | Both endpoints recover in 1-2s |
| Container SIGTERM | Graceful shutdown | Clean exit, no data loss |

## Debugging Commands

```bash
supervisorctl status              # See all processes
supervisorctl tail glm-ocr        # View GLM-OCR logs (port 8002)
supervisorctl tail granite -f     # Follow FastAPI logs (port 7860)
supervisorctl restart glm-ocr     # Restart one service
supervisorctl restart all         # Restart all services

# Check port bindings inside container
netstat -tlnp | grep LISTEN
```<0x0A
## Why supervisord?

- **Lightweight:** No external dependencies
- **Auto-restart:** Built-in process monitoring
- **PID 1 safe:** Proper signal handling for Docker
- **Good logging:** Centralized logs from all services
- **Well-documented:** Mature project

## Critical Design Decisions

1. **Different ports prevent collision:** Each vLLM instance listens on a unique port (8001, 8002)
2. **FastAPI as router:** Uses `@app.api_route()` to match paths and route to correct port
3. **Backward compatible:** `/v1/*` behavior unchanged for existing clients
4. **Independent scalability:** Can easily add more models on ports 8003, 8004, etc.

