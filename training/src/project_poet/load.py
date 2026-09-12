import torch
import torch.nn as nn

from dataclasses import asdict
from pathlib import Path

from project_poet.config import CFG

def save_model(epoch: int, model: nn.Module, opt: torch.optim.Optimizer, vocab_size: int, cfg: CFG, train_loss: list):
    torch.save({
        "epoch": epoch,                    # last finished epoch (resume at epoch+1)
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),     # Adam’s moments; skip this and “resume” is a cold start
        "vocab_size": vocab_size,
        "cfg": asdict(cfg),    # rebuild the net with the same sizes
        "train_loss": train_loss,
    }, cfg.model.model_path)


def load_model(path: str, model: nn.Module, opt: torch.optim.Optimizer, device):
    if not Path(path).exists():
        raise FileNotFoundError("Model ckpt not found!")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    train_loss_log = ckpt["train_loss"]

    return start_epoch, train_loss_log
    