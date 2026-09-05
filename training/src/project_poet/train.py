from pathlib import Path

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config



CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    cfg = get_config(CONFIG_PATH)
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(cfg)

    x, y, pad_mask = next(iter(train_dataloader))
    print(tokenizer.vocab_size, tokenizer.pad_id)
    print(tokenizer.decode(tokenizer.encode("<pad>")))

