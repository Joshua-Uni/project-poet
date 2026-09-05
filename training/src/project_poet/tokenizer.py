from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from project_poet.config import CFG, Tokenizer as TokenizerCfg


PAD_TOKEN = "<pad>"


class PoemTokenizer:
    def __init__(self, hf: Tokenizer, cfg: TokenizerCfg):
        self.hf = hf
        self.cfg = cfg
        self.pad_id = hf.token_to_id(PAD_TOKEN)
        self.title_start_id = hf.token_to_id(cfg.title_start)
        self.title_end_id = hf.token_to_id(cfg.title_end)
        self.poem_start_id = hf.token_to_id(cfg.poem_start)
        self.poem_end_id = hf.token_to_id(cfg.poem_end)

    @property
    def vocab_size(self) -> int:
        return self.hf.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids)

    def encode_example(self, title: str, poem: str) -> list[int]:
        return (
            [self.title_start_id]
            + self.encode(title)
            + [self.title_end_id]
            + [self.poem_start_id]
            + self.encode(poem)
            + [self.poem_end_id]
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.hf.save(path)

    @classmethod
    def load(cls, path: str, cfg: TokenizerCfg) -> "PoemTokenizer":
        return cls(Tokenizer.from_file(path), cfg)


def _train_texts(df):
    for title, poem in zip(df["Title"], df["Poem"]):
        yield str(title)
        yield str(poem)


def get_tokenizer(train_df, cfg: CFG) -> PoemTokenizer:
    specials = [
        PAD_TOKEN,
        cfg.tokenizer.title_start,
        cfg.tokenizer.title_end,
        cfg.tokenizer.poem_start,
        cfg.tokenizer.poem_end,
    ]

    hf = Tokenizer(BPE())
    hf.pre_tokenizer = ByteLevel(add_prefix_space=False)
    hf.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=specials,
    )
    hf.train_from_iterator(_train_texts(train_df), trainer=trainer)

    tok = PoemTokenizer(hf, cfg.tokenizer)
    tok.save(cfg.tokenizer.save_path)
    return tok