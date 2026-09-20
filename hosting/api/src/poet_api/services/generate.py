import json

import torch

from project_poet.config import CFG
from project_poet.engine import generate_text
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer

from poet_api.schemas import GenerateRequest


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_poem(
    req: GenerateRequest,
    model: PoetTransformer,
    tokenizer: PoemTokenizer,
    cfg: CFG,
    device: torch.device,
):
    tokens = (
        [tokenizer.title_start_id]
        + tokenizer.encode(req.title)
        + [tokenizer.title_end_id, tokenizer.poem_start_id]
    )
    response = []

    for _ in range(req.max_tokens):
        next_token = generate_text(
            model, tokens, cfg.model.max_block_size, req.temperature, device
        )
        if next_token == tokenizer.poem_end_id:
            break
        tokens.append(next_token)
        response.append(next_token)
        yield _sse({"poem": tokenizer.decode(response)})

    yield _sse({"done": True, "tokens": len(response)})
