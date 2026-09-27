import torch
import pandas as pd
import logging

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from project_poet.tokenizer import PoemTokenizer
from project_poet.config import CFG

logger = logging.getLogger(__name__)

def get_dataloaders(cfg: CFG, tokenizer: PoemTokenizer | None = None):
    #Get data and split
    raw_df = pd.read_csv(cfg.data.data_path)
    train, val = split_dataframe(raw_df, cfg)


    if tokenizer is None:
        tokenizer = PoemTokenizer.load(cfg.tokenizer.save_path, cfg.tokenizer)
        tokenizer_source = cfg.tokenizer.save_path
    else:
        tokenizer_source = "caller"

    #Build Sequences
    train_seq, val_seq = build_sequences(train, tokenizer, cfg.model.max_block_size), build_sequences(val, tokenizer, cfg.model.max_block_size)

    logger.info(
        "data %s | rows %s | train %s | val %s | batch %s | block %s | vocab %s | tokenizer %s",
        cfg.data.data_path,
        len(raw_df),
        len(train_seq),
        len(val_seq),
        cfg.train.batch_size,
        cfg.model.max_block_size,
        tokenizer.vocab_size,
        tokenizer_source,
    )

    #Make Dataloaders
    collate = make_collate_fn(tokenizer.pad_id)
    train_dataloader = build_dataloader(train_seq, cfg, shuffle=True, collate_fn=collate)
    val_dataloader = build_dataloader(val_seq, cfg, shuffle=False, collate_fn=collate)

    return train_dataloader, val_dataloader, tokenizer



def split_dataframe(df, cfg: CFG):
    val = df.sample(frac=cfg.data.val_ratio, random_state=cfg.data.seed)
    train = df.drop(val.index)

    return train, val



def build_sequences(df, tokenizer: PoemTokenizer, max_block_size):
    seq = [
        tokenizer.encode_example(title, poem)[:max_block_size] #MAX BLOCK SIZE IMPLEMENTED HERE!!!
        for title, poem in zip(df["Title"], df["Poem"])
    ]
    return seq


def build_dataloader(sequences, cfg: CFG, shuffle: bool, collate_fn):
    dataset = PoemDataset(sequences)
    ds_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return ds_loader



class PoemDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [
            torch.tensor(seq, dtype=torch.long) for seq in sequences
        ]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]



def make_collate_fn(pad_id: int):
    def collate_fn(batch):
        x = [seq[:-1] for seq in batch]
        y = [seq[1:] for seq in batch]
        x = pad_sequence(x, batch_first=True, padding_value=pad_id)
        y = pad_sequence(y, batch_first=True, padding_value=-100)

        pad_mask = x == pad_id

        return x, y, pad_mask

    return collate_fn