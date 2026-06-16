FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.20 /uv /usr/local/bin/uv

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

CMD ["sleep", "infinity"]
