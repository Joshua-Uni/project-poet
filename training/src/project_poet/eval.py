from pathlib import Path
import torch
import torch.nn as nn

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config, get_device
from project_poet.model import PoetTransformer
from project_poet.engine import train_one_epoch, eval_one_epoch, generate_text
from project_poet.load import save_model, load_model
from torch.xpu import temperature

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"



if __name__ == "__main__":
    #load model
    cfg = get_config(CONFIG_PATH)
    device = get_device()
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)
    model = PoetTransformer(cfg, tokenizer.vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    load_model(cfg.model.model_path, model, opt, device)

    #Eval Loss
    eval_loss = eval_one_epoch(model, val_dataloader, opt, loss_fn, device)
    print(eval_loss)

    #Sample
    title = "The Dark Night"
    temp = 1.2
    text = [tokenizer.title_start_id] + tokenizer.encode(title) + [tokenizer.title_end_id, tokenizer.poem_start_id]

    for i in range(100):
        next_token = generate_text(model, text, cfg.model.max_block_size, temp, device)
        if next_token == tokenizer.poem_end_id:
            break
        text.append(next_token)

    print(tokenizer.decode(text))

        
