# syntax=docker/dockerfile:1

# CPU-friendly runtime for the Flask/YOLO application.  A GPU-enabled
# deployment can replace this base image and install the corresponding
# CUDA-enabled PyTorch wheel without changing the application code.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV and the video pipeline need these system libraries.  ffmpeg is also
# used for uploaded videos and RTSP capture.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep dependency installation in its own layer so source-only changes do not
# force a full PyTorch/Ultralytics reinstall.
COPY deep_learning_practice_tasks1_2/deep_learning_practice_project/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# The Git repository contains the runnable project one directory below its
# root.  The .dockerignore keeps generated data and reports out of this copy.
COPY deep_learning_practice_tasks1_2/deep_learning_practice_project/ ./

# Runtime output, uploaded files, and model weights should be supplied by the
# deployment (for example, with a mounted volume or YOLO_MODEL_PATH).
RUN mkdir -p /app/outputs /app/runs /models
VOLUME ["/app/outputs", "/models"]

EXPOSE 5000

# Use Flask's CLI here so the container listens on all interfaces (the source
# file's local-development entry point intentionally binds to 127.0.0.1).
CMD ["flask", "--app", "web_app:app", "run", "--host=0.0.0.0", "--port=5000"]
