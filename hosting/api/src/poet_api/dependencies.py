from typing import Annotated
import logging
import torch
from fastapi import Depends, Request, HTTPException

from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer
from project_poet.config import CFG

logger = logging.getLogger(__name__)


def _require(value, name: str):
    if value is None:
        logger.warning("%s not loaded", name)
        raise HTTPException(status_code=503, detail=f"{name} not loaded")
    return value


def get_model(req: Request) -> PoetTransformer:
    model = getattr(req.app.state, "model", None)
    return _require(model, "model")


def get_tokenizer(req: Request) -> PoemTokenizer:
    tokenizer = getattr(req.app.state, "tokenizer", None)
    return _require(tokenizer, "tokenizer")


def get_device(request: Request) -> torch.device:
    device = getattr(request.app.state, "device", None)
    return _require(device, "device")

def get_cfg(request: Request) -> CFG:
    cfg = getattr(request.app.state, "cfg", None)
    return _require(cfg, "config")


ModelDep = Annotated[PoetTransformer, Depends(get_model)]
TokenizerDep = Annotated[PoemTokenizer, Depends(get_tokenizer)]
DeviceDep = Annotated[torch.device, Depends(get_device)]
cfgDep = Annotated[CFG, Depends(get_cfg)]

