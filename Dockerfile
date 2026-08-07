# ---- Base image: PyTorch 2.3 + CUDA 12.1 ----
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Metadata
LABEL maintainer="TriChronos Team"
LABEL version="0.1.0"
LABEL description="TriChronos-0.1B: Ternary-quantised time-series forecasting"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY bitlinear.py .
COPY data_pipeline.py .
COPY model.py .
COPY train.py .
COPY evaluate.py .
COPY publish.py .

# Create directories
RUN mkdir -p checkpoints

# Default environment (override at runtime)
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Expose nothing — this is a training/batch job container, not a web service

# Default entrypoint: training
ENTRYPOINT ["python", "train.py"]

# To evaluate:
#   docker run --gpus all trichronos python evaluate.py --checkpoint checkpoints/model_state.pt
# To publish:
#   docker run --gpus all -e HF_TOKEN=hf_... trichronos python publish.py \
#       --repo-id username/trichronos-0.1b
