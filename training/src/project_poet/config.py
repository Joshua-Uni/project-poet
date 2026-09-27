import yaml
from dataclasses import dataclass
import torch

@dataclass 
class Train:
    resume: bool
    model_path: str
    log_lvl: str
    learning_rate: float
    epochs: int
    batch_size: int


@dataclass 
class Model:
    embedding_dim: int
    max_block_size: int
    n_layers: int
    n_head: int
    head_size: int
    dropout: float


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
    train: Train
    model: Model
    tokenizer: Tokenizer
    data: Data


def cfg_from_dict(raw_cfg: dict) -> CFG:
    return CFG(
        train=Train(**raw_cfg["train"]),
        model=Model(**raw_cfg["model"]),
        tokenizer=Tokenizer(**raw_cfg["tokenizer"]),
        data=Data(**raw_cfg["data"]),
    )


def get_config(config_path):
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    return cfg_from_dict(raw_cfg)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
