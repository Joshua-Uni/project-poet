from fastapi import APIRouter

from poet_api.dependencies import ModelDep, TokenizerDep, DeviceDep, cfgDep
from poet_api.schemas import GenerateRequest, GenerateResponse
from project_poet.engine import generate_text

router = APIRouter(tags=["Generate"])

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, model: ModelDep, tokenizer: TokenizerDep, cfg: cfgDep,  device: DeviceDep):
    response = []
    tokens = [tokenizer.title_start_id] + tokenizer.encode(req.title) + [tokenizer.title_end_id, tokenizer.poem_start_id]

    for i in range(req.max_tokens):
        next_token = generate_text(model, tokens, cfg.model.max_block_size, req.temperature, device)
        if next_token == tokenizer.poem_end_id:
                break
        tokens.append(next_token)
        response.append(next_token)
    


    return {"title": req.title, "poem": tokenizer.decode(response), "tokens": len(response) }
