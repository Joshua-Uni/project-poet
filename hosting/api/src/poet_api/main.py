# Start the server (from hosting/api):
#   uvicorn poet_api.main:app --reload
# Interactive API docs once running:
#   Swagger UI -> http://127.0.0.1:8000/docs
#   ReDoc      -> http://127.0.0.1:8000/redoc

from contextlib import asynccontextmanager
from fastapi import FastAPI

from project_poet.config import get_device
from project_poet.tokenizer import PoemTokenizer

from poet_api.services.load import get_model_and_config
from poet_api.routers import health, generate



@asynccontextmanager
async def lifespan(app: FastAPI):
    #Get model, cfg and tokenizer
    device = get_device()
    model, cfg = get_model_and_config("artifacts/ckpt.pt", device)
    model.eval()
    tokenizer = PoemTokenizer.load(cfg.tokenizer.save_path, cfg.tokenizer)

    app.state.model = model
    app.state.device = device
    app.state.tokenizer = tokenizer
    yield
    app.state.model = None


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(generate.router)