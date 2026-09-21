from pathlib import Path
import torch.nn as nn

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config, get_device, set_seed
from project_poet.engine import eval_one_epoch, generate_text
from project_poet.load import load_inference

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    yaml_cfg = get_config(CONFIG_PATH)
    set_seed(yaml_cfg.data.seed)
    device = get_device()

    model, tokenizer, cfg = load_inference(yaml_cfg.train.model_path, device, yaml_cfg)

    _, val_dataloader, tokenizer = get_dataloaders(cfg, tokenizer)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    eval_loss = eval_one_epoch(model, val_dataloader, loss_fn, device)
    print(eval_loss)

    title = "The Dark Night"
    temp = 1.2
    tokens = [tokenizer.title_start_id] + tokenizer.encode(title) + [tokenizer.title_end_id, tokenizer.poem_start_id]
    response = []
    for i in range(100):
        next_token = generate_text(model, tokens, cfg.model.max_block_size, temp, device)
        if next_token == tokenizer.poem_end_id:
            break
        tokens.append(next_token)
        response.append(next_token)

    print(tokenizer.decode(response))
