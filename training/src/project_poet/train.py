from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config, get_device
from project_poet.model import PoetTransformer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"



def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    opt: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device) -> float:
    model.train()
    total = 0.0
    n_tokens = 0
    for xb, yb, pad_mask in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        logits = model(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        opt.step()

        batch_tokens = int((yb != -100).sum())
        total += loss.item() * batch_tokens
        n_tokens += batch_tokens
        print(loss.item())
    return total / n_tokens



if __name__ == "__main__":
    cfg = get_config(CONFIG_PATH)
    device = get_device()

    train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)
    model = PoetTransformer(cfg, tokenizer.vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    x, y, pad = next(iter(train_dataloader))
    for e in range(3):
        train_loss = train_one_epoch(model, train_dataloader, opt, loss_fn, device)
        print(train_loss)