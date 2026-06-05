# Build on the vast.ai PyTorch image (torch/torchvision/CUDA preinstalled).
# torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, Python 3.12, venv at /venv/main.
FROM vastai/pytorch:2.11.0-cu130-cuda-13.2-mini-py312-2026-04-15

# System libraries:
#  - ffmpeg: video decoding for OpenCV
#  - libgl1 / libglx-mesa0 / libglib2.0-0: required by opencv-contrib-python,
#    which mediapipe pulls in (it links libGL.so.1 even in headless servers).
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglx-mesa0 \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install all project dependencies WITH their transitive deps, but pin
# torch / torchvision / numpy to the versions already shipped in the base image
# so the CUDA-enabled build is never downgraded or replaced.
COPY requirements-docker.txt /tmp/requirements.txt
RUN . /venv/main/bin/activate && \
    pip freeze | grep -iE '^(torch|torchvision|numpy)==' > /tmp/constraints.txt && \
    uv pip install --constraint /tmp/constraints.txt -r /tmp/requirements.txt && \
    uv pip install --constraint /tmp/constraints.txt 'dvc[s3]' && \
    rm -f /tmp/requirements.txt /tmp/constraints.txt

# Source code is mounted (docker-compose) or cloned (vast.ai) at /app.
ENV PYTHONPATH="/app"
WORKDIR /app
