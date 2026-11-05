# Analysis core of FocusCore without the interface: video file in, report out.
#
# Build from the repository root:
#     docker build -f docker/analysis/Dockerfile -t focuscore .
# Run on a lecture recording in the current folder:
#     docker run --rm -v "$PWD":/data focuscore /data/lecture.mp4
# Add -e OPENAI_API_KEY=sk-... to transcribe the critical moments.

FROM python:3.10-slim

ARG MODEL_URL=https://github.com/Ayanami-Rei1/FocusCore/releases/download/v1.0/engagement_slow_r50.ts

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# The pip bundled with the base image rejects some current wheels, so it is updated first.
RUN pip install --no-cache-dir --upgrade pip

# torch is pinned to the version the model was tested with in the application.
# The +cpu builds come from the PyTorch index, everything else from PyPI.
RUN pip install --no-cache-dir "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

RUN mkdir -p app/models && \
    python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], 'app/models/engagement_slow_r50.ts')" "$MODEL_URL"

COPY app/__init__.py app/cli.py app/
COPY app/core app/core

ENTRYPOINT ["python", "-m", "app.cli"]
CMD ["--help"]
