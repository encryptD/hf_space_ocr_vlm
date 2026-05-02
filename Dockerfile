FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# HF Spaces runs containers as uid 1000 — create the user and home dir.
RUN useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY --chown=user src/ ./src/
COPY --chown=user granite4_vision.py ./
COPY --chown=user start_granite4_vision_server.py ./

# Switch to non-root user for runtime (required by HF Spaces).
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
