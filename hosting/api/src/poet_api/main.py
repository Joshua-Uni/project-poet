# Optional env (default checkpoint is artifacts/ckpt.pt under the repo root, log level INFO):
#   POET_CKPT_PATH=... POET_LOG_LEVEL=INFO uvicorn poet_api.main:app --reload
# Interactive API docs once running:
#   Swagger UI -> http://127.0.0.1:8000/docs
#   ReDoc      -> http://127.0.0.1:8000/redoc

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from project_poet.config import get_device, repo_root

from poet_api.logging import setup_logging
from poet_api.routers import generate, health
from poet_api.services.load import get_model_and_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(os.getenv("POET_LOG_LEVEL", "INFO"))

    device = get_device()
    # Same root as training. A relative POET_CKPT_PATH is still used as given.
    default_ckpt = str(repo_root() / "artifacts" / "ckpt.pt")
    ckpt_path = os.getenv("POET_CKPT_PATH", default_ckpt)
    model, tokenizer, cfg = get_model_and_config(ckpt_path, device)
    model.eval()
    logger.info("api ready | block %s", cfg.model.max_block_size)

    app.state.model = model
    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.cfg = cfg
    yield
    logger.info("api shutdown")
    app.state.model = None


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(generate.router)