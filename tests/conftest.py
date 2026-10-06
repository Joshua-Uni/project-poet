"""Tiny in-memory tokenizer and transformer shared by the suite."""

from types import SimpleNamespace

import pytest
from project_poet.config import Model as ModelCfg
from project_poet.config import Tokenizer as TokenizerCfg
from project_poet.model import PoetTransformer
from project_poet.tokenizer import GENRES, PAD_TOKEN, PoemTokenizer
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@pytest.fixture
def tokenizer():
    cfg = TokenizerCfg(
        vocab_size=100,
        save_path="",
        title_start="<title>",
        title_end="</title>",
        poem_start="<poem>",
        poem_end="</poem>",
    )
    specials = [
        PAD_TOKEN,
        cfg.title_start,
        cfg.title_end,
        cfg.poem_start,
        cfg.poem_end,
        *GENRES.values(),
    ]
    hf = Tokenizer(BPE())
    hf.pre_tokenizer = ByteLevel(add_prefix_space=False)
    hf.decoder = ByteLevelDecoder()
    hf.train_from_iterator(
        ["Night", "the dark sea"],
        trainer=BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=specials),
    )
    return PoemTokenizer(hf, cfg)


@pytest.fixture
def tiny_model():
    # head_size * n_head matches embedding_dim so the attention residual can add.
    model_cfg = ModelCfg(
        embedding_dim=8,
        max_block_size=16,
        n_layers=1,
        n_head=2,
        head_size=4,
        dropout=0.0,
    )
    return PoetTransformer(SimpleNamespace(model=model_cfg), vocab_size=32)
