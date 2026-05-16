FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Use stable system Python 3.10.12 — Ubuntu 22.04's python3.11 package is
# 3.11.0~rc1 (a pre-release) which has known segfault bugs in
# _PyEval_EvalFrameDefault that crash vLLM's encoder cache profiling.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as uid 1000 — create the user and home dir.
RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# GLM-OCR requires Transformers 5.x model definitions (glm_ocr), while vLLM
# 0.19.0 currently installs a 4.x release transitively.
RUN python3 -m pip install --no-cache-dir --no-deps \
    transformers==5.1.0

COPY --chown=user src/ ./src/
COPY --chown=user granite4_vision.py ./
COPY --chown=user start_granite4_vision_server.py ./

# Switch to non-root user for runtime (required by HF Spaces).
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]
