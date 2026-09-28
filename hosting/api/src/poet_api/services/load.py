import torch
from project_poet.config import CFG
from project_poet.load import load_inference
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer


def get_model_and_config(path: str, device: torch.device) -> tuple[PoetTransformer, PoemTokenizer, CFG]:
    return load_inference(path, device)
