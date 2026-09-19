# Optional paths (defaults: artifacts/ckpt.pt and tokenizer.json next to the ckpt):
#   POET_CKPT_PATH=... POET_TOKENIZER_PATH=... uvicorn poet_api.main:app --reload
# Interactive API docs once running:
#   Swagger UI -> http://127.0.0.1:8000/docs
#   ReDoc      -> http://127.0.0.1:8000/redoc

from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
from pathlib import Path

from project_poet.config import get_device
from project_poet.tokenizer import PoemTokenizer

from poet_api.services.load import get_model_and_config
from poet_api.routers import health, generate



@asynccontextmanager
async def lifespan(app: FastAPI):
    #Get model, cfg and tokenizer
    device = get_device()
    ckpt_path = os.getenv("POET_CKPT_PATH", "artifacts/ckpt.pt")

    model, cfg = get_model_and_config(ckpt_path, device)
    model.eval()

    default_tokenizer = str(Path(ckpt_path).parent / "tokenizer.json")
    tokenizer_path = os.getenv("POET_TOKENIZER_PATH", default_tokenizer)
    tokenizer = PoemTokenizer.load(tokenizer_path, cfg.tokenizer)

    app.state.model = model
    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.cfg = cfg
    yield
    app.state.model = None


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(generate.router)