import logging
from pathlib import Path

import torch
import torch.nn as nn

from project_poet.config import get_config, get_device, set_seed
from project_poet.dataloader import get_dataloaders
from project_poet.engine import eval_one_epoch, train_one_epoch
from project_poet.load import load_training, save_training
from project_poet.logging import setup_logging
from project_poet.model import PoetTransformer

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    yaml_cfg = get_config(CONFIG_PATH)
    setup_logging(yaml_cfg.train.log_lvl)

    set_seed(yaml_cfg.data.seed)
    device = get_device()

    logger.info(
        "train | device %s | resume %s | epochs %s | lr %s | max_norm %s | batch %s",
        device,
        yaml_cfg.train.resume,
        yaml_cfg.train.epochs,
        yaml_cfg.train.learning_rate,
        yaml_cfg.train.max_norm,
        yaml_cfg.train.batch_size,
    )

    if yaml_cfg.train.resume:
        model, opt, tokenizer, cfg, start_epoch, train_loss_log, eval_loss_log = load_training(
            yaml_cfg, device
        )
        train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg, tokenizer)
    else:
        cfg = yaml_cfg
        train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)
        model = PoetTransformer(cfg, tokenizer.vocab_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
        start_epoch = 0
        train_loss_log = []
        eval_loss_log = []
        logger.info("fresh model | vocab %s | layers %s | emb %s", tokenizer.vocab_size, cfg.model.n_layers, cfg.model.embedding_dim)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(start_epoch, cfg.train.epochs):
        train_loss = train_one_epoch(model, cfg.train, train_dataloader, opt, loss_fn, device)
        train_loss_log.append(train_loss)

        eval_loss = eval_one_epoch(model, val_dataloader, loss_fn, device)
        eval_loss_log.append(eval_loss)
        logger.info("epoch %s | train %.4f | val %.4f", epoch, train_loss, eval_loss)

        save_training(epoch, model, opt, cfg, tokenizer, train_loss_log, eval_loss_log)
