FROM vastai/base-image:cuda-12.8.1-auto

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN . /venv/main/bin/activate && \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements-docker.txt /tmp/requirements.txt
RUN . /venv/main/bin/activate && \
    uv pip install --no-deps -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN . /venv/main/bin/activate && \
    uv pip install dvc[s3]

ENV PYTHONPATH="/workspace/focus-core:${PYTHONPATH}"

WORKDIR /workspace
