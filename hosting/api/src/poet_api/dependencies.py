from typing import Annotated
import torch
from fastapi import Depends, Request

from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer





def get_model(req: Request) -> PoetTransformer:
    model = getattr(req.app.state, "model", None)
    if model is None:
        raise ValueError("hi", model)

    return model


def get_tokenizer(req: Request) -> PoemTokenizer:
    tokenizer = getattr(req.app.state, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("ayy")
    return tokenizer


def get_device(request: Request) -> torch.device:
    device = getattr(request.app.state, "device", None)
    if device is None:
        raise ValueError("gotta fix this")
    return device


ModelDep = Annotated[PoetTransformer, Depends(get_model)]
TokenizerDep = Annotated[PoemTokenizer, Depends(get_tokenizer)]
DeviceDep = Annotated[torch.device, Depends(get_device)]

