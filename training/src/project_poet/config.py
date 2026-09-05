import yaml
from dataclasses import dataclass

@dataclass 
class Model:
    learning_rate: float
    batch_size: int


@dataclass
class Tokenizer:
    vocab_size: int
    save_path: str
    title_start: str
    title_end: str
    poem_start: str
    poem_end: str


@dataclass
class Data:
    data_path: str
    seed: int
    val_ratio: float


@dataclass
class CFG:
    model: Model
    tokenizer: Tokenizer
    data: Data


def get_config(config_path):
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    
    cfg = CFG(model=Model(**raw_cfg["model"]),
              tokenizer=Tokenizer(**raw_cfg["tokenizer"]),
              data=Data(**raw_cfg["data"]))
    
    return cfg