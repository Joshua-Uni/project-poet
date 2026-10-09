# Project Poet: one container serving the API and the website (FastAPI + uvicorn).
#
# Build (checkpoint must exist at artifacts/ckpt.pt):
#   docker build --platform linux/amd64 -t project-poet .
# Run:
#   docker run --rm -p 8000:8000 project-poet     -> http://127.0.0.1:8000

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The repo layout is kept as /app/... because repo_root() finds artifacts/ and
# hosting/web relative to the source files (hence the editable installs below).
WORKDIR /app

# CPU-only torch: the default Linux wheel bundles CUDA libraries we never use.
# Installed first so this large layer is cached across code changes.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Both packages, editable, so project_poet stays under /app/training/src.
COPY training/pyproject.toml training/pyproject.toml
COPY training/src training/src
COPY hosting/api/pyproject.toml hosting/api/pyproject.toml
COPY hosting/api/src hosting/api/src
RUN pip install -e training -e hosting/api

# Site and model. Last, since they change most often. The checkpoint also
# carries the tokenizer and model config, so nothing else is needed.
COPY hosting/web hosting/web
COPY artifacts/ckpt.pt artifacts/ckpt.pt

RUN useradd --create-home poet
USER poet

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Single worker: each worker would load its own copy of the model.
CMD ["uvicorn", "poet_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
