from pathlib import Path
import torch
import torch.nn as nn

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config, get_device
from project_poet.model import PoetTransformer
from project_poet.engine import train_one_epoch

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    cfg = get_config(CONFIG_PATH)
    device = get_device()

    train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)
    model = PoetTransformer(cfg, tokenizer.vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    for e in range(3):
        train_loss = train_one_epoch(model, train_dataloader, opt, loss_fn, device)
        print(train_loss)