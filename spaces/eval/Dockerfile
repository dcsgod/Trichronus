# ---- Eval Space Dockerfile (CPU-only) ----
FROM python:3.11-slim

LABEL maintainer="iravikr"
LABEL description="TriChronos-0.1B Evaluation Space"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CPU torch — much smaller image than CUDA)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch>=2.3.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY bitlinear.py data_pipeline.py model.py evaluate.py ./
COPY app.py .

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
