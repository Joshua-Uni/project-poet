import json
import logging
import time
from collections.abc import Iterator

import torch
from poet_api.schemas import GenerateRequest
from project_poet.config import CFG
from project_poet.engine import generate_text
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer

logger = logging.getLogger(__name__)


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_poem(
    req: GenerateRequest,
    model: PoetTransformer,
    tokenizer: PoemTokenizer,
    cfg: CFG,
    device: torch.device,
) -> Iterator[str]:
    started = time.perf_counter()
    logger.info(
        "generate stream | title %r | temp %s | max_tokens %s",
        req.title,
        req.temperature,
        req.max_tokens,
    )
    tokens = (
        [tokenizer.title_start_id]
        + tokenizer.encode(req.title)
        + [tokenizer.title_end_id, tokenizer.poem_start_id]
    )
    response = []

    for _ in range(req.max_tokens):
        next_token = generate_text(
            model, tokens, cfg.model.max_block_size, req.temperature, device, req.top_k, req.top_p
        )
        if next_token == tokenizer.poem_end_id:
            break
        tokens.append(next_token)
        response.append(next_token)
        yield _sse({"poem": tokenizer.decode(response)})

    poem = tokenizer.decode(response)
    logger.info("generate stream done | tokens %s | %.2fs", len(response), time.perf_counter() - started)
    logger.debug("generate stream | poem\n%s", poem)
    yield _sse({"done": True, "tokens": len(response)})
