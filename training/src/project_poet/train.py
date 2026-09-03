from pathlib import Path

from project_poet.dataloader import get_dataloaders
from project_poet.config import get_config



CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


if __name__ == "__main__":
    cfg = get_config(CONFIG_PATH)
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    x, y, pad_mask = next(iter(train_dataloader))
    print(cfg.model.learning_rate)
    print(x.shape)