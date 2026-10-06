# Dockerfile for the Image -> Story production pipeline.
# Multi-stage build to keep the final image small.

FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --user .

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY configs/ ./configs/
COPY benchmarks/ ./benchmarks/

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# The pipeline runs offline by default; models must be pre-cached.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# One entry point for the production system.
ENTRYPOINT ["image-story"]
CMD ["--help"]