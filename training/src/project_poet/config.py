from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass 
class Train:
    resume: bool
    model_path: str
    log_lvl: str
    learning_rate: float
    weight_decay: float
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


def repo_root() -> Path:
    # config.py -> project_poet -> src -> training -> repo root ("Project Poet")
    # Anchored to this file, so it does not change with the shell's current directory.
    return Path(__file__).resolve().parents[3]


def _abs_from_root(path: str) -> str:
    # Yaml paths like "artifacts/ckpt.pt" are relative to the repo root.
    # An absolute path (Colab, Drive) is kept as written.
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(repo_root() / p)


def get_config(config_path: str | Path) -> CFG:
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = cfg_from_dict(raw_cfg)
    cfg.train.model_path = _abs_from_root(cfg.train.model_path)
    cfg.tokenizer.save_path = _abs_from_root(cfg.tokenizer.save_path)
    cfg.data.data_path = _abs_from_root(cfg.data.data_path)
    return cfg


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
