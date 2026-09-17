import torch
from pathlib import Path

from project_poet.config import CFG, Model, Tokenizer, Data
from project_poet.model import PoetTransformer


def get_model_and_config(path: str, device):
    if not Path(path).exists():
        raise FileNotFoundError("Model ckpt not found!")

    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["cfg"]
    cfg = CFG(
        model=Model(**cfg_dict["model"]),
        tokenizer=Tokenizer(**cfg_dict["tokenizer"]),
        data=Data(**cfg_dict["data"]),
    )

    model = PoetTransformer(cfg, ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device)

    return model, cfg
