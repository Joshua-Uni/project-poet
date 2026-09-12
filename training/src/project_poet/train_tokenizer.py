import pandas as pd
from pathlib import Path

from project_poet.tokenizer import train_tokenizer
from project_poet.config import get_config
from project_poet.dataloader import split_dataframe


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

if __name__ == "__main__":
    #Get data and split
    cfg = get_config(CONFIG_PATH)

    raw_df = pd.read_csv(cfg.data.data_path)
    train, val = split_dataframe(raw_df, cfg)

    tokenizer = train_tokenizer(train, cfg)
