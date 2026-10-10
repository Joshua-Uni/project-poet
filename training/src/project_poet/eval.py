import logging
from pathlib import Path

import torch.nn as nn

from project_poet.config import get_config, get_device, set_seed
from project_poet.dataloader import get_dataloaders
from project_poet.engine import eval_one_epoch, generate_text
from project_poet.load import load_inference
from project_poet.logging import setup_logging

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    yaml_cfg = get_config(CONFIG_PATH)
    setup_logging(yaml_cfg.train.log_lvl)
    set_seed(yaml_cfg.data.seed)
    device = get_device()
    logger.info("eval | device %s | ckpt %s", device, yaml_cfg.train.model_path)

    model, tokenizer, cfg = load_inference(yaml_cfg.train.model_path, device, yaml_cfg)

    _, val_dataloader, tokenizer = get_dataloaders(cfg, tokenizer)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    eval_loss = eval_one_epoch(model, val_dataloader, loss_fn, device)
    logger.info("val loss %.4f", eval_loss)

    title = "The Dark Night"
    temp = 1.2
    top_k = 30
    top_p = 0.9
    genre = "Love"
    logger.info(
        "sample | title %r | genre %s | temp %s | top_k %s | top_p %s",
        title,
        genre,
        temp,
        top_k,
        top_p,
    )
    tokens = tokenizer.prompt_ids(title, genre)
    response = []
    for i in range(100):
        next_token = generate_text(
            model, tokens, cfg.model.max_block_size, temp, device, top_k, top_p
        )
        if next_token == tokenizer.poem_end_id:
            break
        tokens.append(next_token)
        response.append(next_token)

    logger.info("sample | tokens %s\n%s", len(response), tokenizer.decode(response))
