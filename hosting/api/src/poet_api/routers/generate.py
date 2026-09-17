from fastapi import APIRouter

from poet_api.dependencies import ModelDep

router = APIRouter(tags=["Generate"])

@router.get("/generate")
def generate(model: ModelDep):

    return {"Info": model.blocks}