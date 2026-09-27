import logging
import torch
import torch.nn as nn

from dataclasses import asdict, replace
from pathlib import Path
from tokenizers import Tokenizer as HfTokenizer

from project_poet.config import CFG, cfg_from_dict
from project_poet.model import PoetTransformer
from project_poet.tokenizer import PoemTokenizer

logger = logging.getLogger(__name__)


def apply_yaml_run(cfg: CFG, yaml_cfg: CFG) -> CFG:
    #Take a cfg that you loaded from checkpoint and replaces its cfg.data and cfg.train
    #with the yaml_cfg (the current config.yaml in the directory)
    #config.yaml controls the data, model_path, epochs, batch_size e.t.c. Checkpoint cfg controls model
    return replace(
        cfg,
        data=yaml_cfg.data,
        train=yaml_cfg.train,
    )


def save_training(epoch: int, model: nn.Module, opt: torch.optim.Optimizer, cfg: CFG, tokenizer: PoemTokenizer, train_loss: list, eval_loss: list):
    torch.save({
        "epoch": epoch,                    # last finished epoch (resume at epoch+1)
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),     # AdamW’s moments; skip this and “resume” is a cold start
        "vocab_size": tokenizer.vocab_size,
        "cfg": asdict(cfg),    # rebuild the net with the same sizes
        "tokenizer": tokenizer.hf.to_str(),
        "train_loss": train_loss,
        "eval_loss": eval_loss
    }, cfg.train.model_path)
    logger.info("saved checkpoint | epoch %s | %s", epoch, cfg.train.model_path)


def _load_ckpt(path: str, device):
    #Returns the loaded Checkpoint
    if not Path(path).exists():
        logger.error("checkpoint not found: %s", path)
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
    #Loads only the model, tokenizer and cfg
    #Can pass in a yaml_cfg to update your cfg's data and train updated
    #e.g. this is good for eval since you'll use dataloaders with the yaml_cfg batchsize
    #unneccesary for api since only care about model
    ckpt = _load_ckpt(path, device)
    model, tokenizer, cfg = _rebuild(ckpt, device)
    if yaml_cfg is not None:
        cfg = apply_yaml_run(cfg, yaml_cfg)
    logger.info("loaded checkpoint | %s | vocab %s | device %s", path, tokenizer.vocab_size, device)
    return model, tokenizer, cfg


def load_training(yaml_cfg: CFG, device):
    #Loads everything needed for training (everything in a checkpoint)
    #Replaces checkpoint configs .train and .data with yaml_cfg
    ckpt = _load_ckpt(yaml_cfg.train.model_path, device)
    model, tokenizer, cfg = _rebuild(ckpt, device)
    cfg = apply_yaml_run(cfg, yaml_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=0.01)
    opt.load_state_dict(ckpt["optimizer"])
    for g in opt.param_groups:
        g["lr"] = cfg.train.learning_rate
    start_epoch = ckpt["epoch"] + 1
    logger.info(
        "resume | %s | next epoch %s | lr %s | vocab %s",
        yaml_cfg.train.model_path,
        start_epoch,
        cfg.train.learning_rate,
        tokenizer.vocab_size,
    )
    return model, opt, tokenizer, cfg, start_epoch, ckpt["train_loss"], ckpt["eval_loss"]
