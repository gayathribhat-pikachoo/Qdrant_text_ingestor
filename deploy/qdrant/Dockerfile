FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface

COPY deploy/qdrant/requirements-ingestor.txt /tmp/requirements-ingestor.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements-ingestor.txt

COPY . .

CMD ["uvicorn", "fastembed:app", "--host", "0.0.0.0", "--port", "8000"]