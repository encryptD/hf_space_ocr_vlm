FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

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

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
