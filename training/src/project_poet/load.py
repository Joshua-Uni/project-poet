import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tokenizers import Tokenizer as HfTokenizer
from torch.optim.lr_scheduler import LambdaLR

from project_poet.config import CFG, cfg_from_dict
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer

logger = logging.getLogger(__name__)


def apply_yaml_run(cfg: CFG, yaml_cfg: CFG) -> CFG:
    # Take a cfg that you loaded from checkpoint and replaces its cfg.data and cfg.train
    # with the yaml_cfg (the current config.yaml in the directory)
    # config.yaml controls the data, model_path, epochs, batch_size e.t.c. Checkpoint cfg controls model
    return replace(
        cfg,
        data=yaml_cfg.data,
        train=yaml_cfg.train,
    )


def save_training(
    epoch: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: LambdaLR,
    cfg: CFG,
    tokenizer: PoemTokenizer,
    train_loss: list[float],
    eval_loss: list[float],
    path: str | None = None,
) -> None:
    # path defaults to the resume checkpoint. Training also passes best_path.
    out = cfg.train.model_path if path is None else path
    torch.save(
        {
            "epoch": epoch,  # last finished epoch (resume at epoch+1)
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),  # AdamW’s moments; skip this and “resume” is a cold start
            "scheduler": scheduler.state_dict(),  # last_epoch is the batch count so far
            "vocab_size": tokenizer.vocab_size,
            "cfg": asdict(cfg),  # rebuild the net with the same sizes
            "tokenizer": tokenizer.hf.to_str(),
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        },
        out,
    )
    logger.info("saved checkpoint | epoch %s | %s", epoch, out)


def _load_ckpt(path: str, device: torch.device) -> dict[str, Any]:
    # Returns the loaded Checkpoint
    if not Path(path).exists():
        logger.error("checkpoint not found: %s", path)
        raise FileNotFoundError("Model ckpt not found!")
    return torch.load(path, map_location=device)


def _rebuild(
    ckpt: dict[str, Any], device: torch.device
) -> tuple[PoetTransformer, PoemTokenizer, CFG]:
    # Rebuilds CFG, Tokenizer and Model from a Checkpoint
    cfg = cfg_from_dict(ckpt["cfg"])
    tokenizer = PoemTokenizer(HfTokenizer.from_str(ckpt["tokenizer"]), cfg.tokenizer)
    model = PoetTransformer(cfg, ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, tokenizer, cfg


def load_inference(
    path: str, device: torch.device, yaml_cfg: CFG | None = None
) -> tuple[PoetTransformer, PoemTokenizer, CFG]:
    # Loads only the model, tokenizer and cfg
    # Can pass in a yaml_cfg to update your cfg's data and train updated
    # e.g. this is good for eval since you'll use dataloaders with the yaml_cfg batchsize
    # unneccesary for api since only care about model
    ckpt = _load_ckpt(path, device)
    model, tokenizer, cfg = _rebuild(ckpt, device)
    if yaml_cfg is not None:
        cfg = apply_yaml_run(cfg, yaml_cfg)
    logger.info("loaded checkpoint | %s | vocab %s | device %s", path, tokenizer.vocab_size, device)
    return model, tokenizer, cfg


def load_training(
    yaml_cfg: CFG, device: torch.device
) -> tuple[
    PoetTransformer,
    torch.optim.Optimizer,
    PoemTokenizer,
    CFG,
    int,
    int,
    list[float],
    list[float],
]:
    # Loads everything needed for training (everything in a checkpoint)
    # Replaces checkpoint configs .train and .data with yaml_cfg
    ckpt = _load_ckpt(yaml_cfg.train.model_path, device)
    model, tokenizer, cfg = _rebuild(ckpt, device)
    cfg = apply_yaml_run(cfg, yaml_cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    opt.load_state_dict(ckpt["optimizer"])
    # The lr is left to the scheduler, which train.py rebuilds at start_step.
    for g in opt.param_groups:
        g["weight_decay"] = cfg.train.weight_decay
    start_epoch = ckpt["epoch"] + 1
    start_step = ckpt["scheduler"]["last_epoch"]

    logger.info(
        "resume | %s | next epoch %s | peak lr %s | weight_decay %s | max_norm %s | vocab %s",
        yaml_cfg.train.model_path,
        start_epoch,
        cfg.train.learning_rate,
        cfg.train.weight_decay,
        cfg.train.max_norm,
        tokenizer.vocab_size,
    )
    return (
        model,
        opt,
        tokenizer,
        cfg,
        start_epoch,
        start_step,
        ckpt["train_loss"],
        ckpt["eval_loss"],
    )
