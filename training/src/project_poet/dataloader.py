import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from project_poet.tokenizer import get_tokenizer
from project_poet.config import CFG


PAD_ID = 0 #Eventually get this from somewhere else

def get_dataloaders(cfg: CFG):
    #Get data and split
    raw_df = pd.read_csv(cfg.data.data_path)
    train, val = split_dataframe(raw_df, cfg)

    #Get Encoder
    encode, decode = get_tokenizer(train)

    #Build Sequences
    train_seq, val_seq = build_sequences(train, encode, cfg), build_sequences(val, encode, cfg)
    
    #Make Dataloaders
    train_dataloader, val_dataloader = build_dataloader(train_seq, cfg, shuffle=True), build_dataloader(val_seq, cfg, shuffle=False)

    return train_dataloader, val_dataloader



def split_dataframe(df, cfg: CFG):
    val = df.sample(frac=cfg.data.val_ratio, random_state=cfg.data.seed)
    train = df.drop(val.index)

    return train, val



def build_sequences(df, encode, cfg: CFG):
    sequences = []
    for title, poem in zip(df["Title"], df["Poem"]):
        text = (
                f"{cfg.tokenizer.title_start}{title}{cfg.tokenizer.title_end}"
                f"{cfg.tokenizer.poem_start}{poem}{cfg.tokenizer.poem_end}"
            )
        sequences.append(encode(text))
    return sequences



def build_dataloader(sequences, cfg, shuffle: bool):
    dataset = PoemDataset(sequences)
    ds_loader = DataLoader(dataset, batch_size = cfg.model.batch_size, shuffle=shuffle, collate_fn=collate_fn)
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



def collate_fn(batch):
    x = [seq[:-1] for seq in batch]
    y = [seq[1:] for seq in batch]

    x = pad_sequence(x, batch_first=True, padding_value=PAD_ID)
    y = pad_sequence(y, batch_first=True, padding_value=-100)

    pad_mask = (x == PAD_ID)

    return x, y, pad_mask