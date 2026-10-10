from fastapi import APIRouter
from poet_api.dependencies import ModelDep, TokenizerDep

router = APIRouter(tags=["Health"])


@router.get("/health")
def get_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def get_ready(model: ModelDep, tokenizer: TokenizerDep) -> dict[str, str]:
    return {"status": "ready"}
