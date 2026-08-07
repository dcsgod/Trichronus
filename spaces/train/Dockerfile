# ---- Training Space Dockerfile (A100 + Gradio UI) ----
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="iravikr"
LABEL description="TriChronos-0.1B Training Space"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (includes gradio for the monitoring UI)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY bitlinear.py data_pipeline.py model.py train.py evaluate.py publish.py ./
COPY app.py .

# Directories
RUN mkdir -p checkpoints

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

# Entry point: Gradio monitoring UI (which manages train.py as a subprocess)
CMD ["python", "app.py"]
