import logging
from typing import Annotated, TypeVar

import torch
from fastapi import Depends, HTTPException, Request
from project_poet.config import CFG
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _require(value: T | None, name: str) -> T:
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

