import torch
import torch.nn as nn

from dataclasses import asdict, replace
from pathlib import Path
from tokenizers import Tokenizer as HfTokenizer

from project_poet.config import CFG, cfg_from_dict
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer

# yaml owns the run; the checkpoint owns architecture + tokenizer
_YAML_MODEL_FIELDS = ("model_path", "resume", "epochs", "batch_size")


def apply_yaml_run(cfg: CFG, yaml_cfg: CFG) -> CFG:
    #Take a cfg that you loaded from checkpoint and replaces its cfg.data and cfg.model._YAML_MODEL_FIELDS
    #with the yaml_cfg (the current config.yaml in the directory)
    #config.yaml controls the data, model_path, epochs, batch_size e.t.c. Checkpoint cfg controls model
    return replace(
        cfg,
        data=yaml_cfg.data,
        model=replace(cfg.model, **{name: getattr(yaml_cfg.model, name) for name in _YAML_MODEL_FIELDS}),
    )


def save_training(epoch: int, model: nn.Module, opt: torch.optim.Optimizer, cfg: CFG, tokenizer: PoemTokenizer, train_loss: list, eval_loss: list):
    torch.save({
        "epoch": epoch,                    # last finished epoch (resume at epoch+1)
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),     # Adam’s moments; skip this and “resume” is a cold start
        "vocab_size": tokenizer.vocab_size,
        "cfg": asdict(cfg),    # rebuild the net with the same sizes
        "tokenizer": tokenizer.hf.to_str(),
        "train_loss": train_loss,
        "eval_loss": eval_loss
    }, cfg.model.model_path)


def _load_ckpt(path: str, device):
    #Returns the loaded Checkpoint
    if not Path(path).exists():
        raise FileNotFoundError("Model ckpt not found!")
    return torch.load(path, map_location=device)


def _rebuild(ckpt, device):
    #Rebuilds CFG, Tokenizer and Model from a Checkpoint
    cfg = cfg_from_dict(ckpt["cfg"])
    tokenizer = PoemTokenizer(HfTokenizer.from_str(ckpt["tokenizer"]), cfg.tokenizer)
    model = PoetTransformer(cfg, ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, tokenizer, cfg


def load_inference(path: str, device, yaml_cfg: CFG | None = None):
    ckpt = _load_ckpt(path, device)
    model, tokenizer, cfg = _rebuild(ckpt, device)
    if yaml_cfg is not None:
        cfg = apply_yaml_run(cfg, yaml_cfg)
    return model, tokenizer, cfg


def load_training(yaml_cfg: CFG, device):
    ckpt = _load_ckpt(yaml_cfg.model.model_path, device)
    model, tokenizer, cfg = _rebuild(ckpt, device)
    cfg = apply_yaml_run(cfg, yaml_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    opt.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    return model, opt, tokenizer, cfg, start_epoch, ckpt["train_loss"], ckpt["eval_loss"]
