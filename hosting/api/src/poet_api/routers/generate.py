import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from poet_api.dependencies import ModelDep, TokenizerDep, DeviceDep, cfgDep
from poet_api.schemas import GenerateRequest, GenerateResponse
from poet_api.services.generate import stream_poem
from project_poet.engine import generate_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Generate"])

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, model: ModelDep, tokenizer: TokenizerDep, cfg: cfgDep,  device: DeviceDep):
    started = time.perf_counter()
    logger.info("generate | title %r | temp %s | max_tokens %s", req.title, req.temperature, req.max_tokens)
    response = []
    tokens = [tokenizer.title_start_id] + tokenizer.encode(req.title) + [tokenizer.title_end_id, tokenizer.poem_start_id]

    for i in range(req.max_tokens):
        next_token = generate_text(model, tokens, cfg.model.max_block_size, req.temperature, device)
        if next_token == tokenizer.poem_end_id:
                break
        tokens.append(next_token)
        response.append(next_token)

    poem = tokenizer.decode(response)
    logger.info("generate done | tokens %s | %.2fs", len(response), time.perf_counter() - started)
    logger.debug("generate | poem\n%s", poem)
    return {"title": req.title, "poem": poem, "tokens": len(response) }



@router.post("/generate/stream")
def generate_stream(req: GenerateRequest, model: ModelDep, tokenizer: TokenizerDep, cfg: cfgDep,  device: DeviceDep):
    return StreamingResponse(
        stream_poem(req, model, tokenizer, cfg, device),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

