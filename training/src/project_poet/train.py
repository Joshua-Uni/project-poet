from pathlib import Path
import torch.nn as nn
import torch

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config, get_device, set_seed
from project_poet.model import PoetTransformer
from project_poet.engine import train_one_epoch, eval_one_epoch
from project_poet.load import save_training, load_training

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    yaml_cfg = get_config(CONFIG_PATH)
    set_seed(yaml_cfg.data.seed)
    device = get_device()

    if yaml_cfg.train.resume:
        model, opt, tokenizer, cfg, start_epoch, train_loss_log, eval_loss_log = load_training(
            yaml_cfg, device
        )
        train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg, tokenizer)
    else:
        cfg = yaml_cfg
        train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)
        model = PoetTransformer(cfg, tokenizer.vocab_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=0.01)
        start_epoch = 0
        train_loss_log = []
        eval_loss_log = []

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loss = train_one_epoch(model, train_dataloader, opt, loss_fn, device)
        train_loss_log.append(train_loss)
        print(train_loss)

        eval_loss = eval_one_epoch(model, val_dataloader, loss_fn, device)
        eval_loss_log.append(eval_loss)
        print(eval_loss)
        print("\n")

        save_training(epoch, model, opt, cfg, tokenizer, train_loss_log, eval_loss_log)
