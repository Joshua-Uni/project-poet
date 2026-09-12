from pathlib import Path
import torch
import torch.nn as nn

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config, get_device
from project_poet.model import PoetTransformer
from project_poet.engine import train_one_epoch
from project_poet.load import save_model, load_model

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    cfg = get_config(CONFIG_PATH)
    device = get_device()
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)
    model = PoetTransformer(cfg, tokenizer.vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    start_epoch = 0
    train_loss_log = []
    if cfg.model.resume:
        start_epoch, train_loss_log = load_model(cfg.model.model_path, model, opt, device)


    for epoch in range(start_epoch, cfg.model.epochs):
        train_loss = train_one_epoch(model, train_dataloader, opt, loss_fn, device)
        train_loss_log.append(train_loss)
        print(train_loss)

        #save checkpoint
        save_model(epoch, model, opt, tokenizer.vocab_size, cfg, train_loss_log)
