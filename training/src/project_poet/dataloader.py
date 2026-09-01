import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from project_poet.tokenizer import get_tokenizer
from project_poet.config import CFG


PAD_ID = 0 #Eventually get this from somewhere else

def get_dataloaders(cfg: CFG):

    raw_data = pd.read_csv(cfg.data.data_path)
    encode, decode = get_tokenizer(raw_data)

    #Tokenize raw data and put into lists
    sequences = []
    for title, poem in zip(raw_data["Title"], raw_data["Poem"]):
        text = (
                f"{cfg.tokenizer.title_start}{title}{cfg.tokenizer.title_end}"
                f"{cfg.tokenizer.poem_start}{poem}{cfg.tokenizer.poem_end}"
            )

        sequences.append(encode(text))
    
    #make dataset
    dataset = PoemDataset(sequences)

    #make dataloader
    ds_loader = DataLoader(dataset, batch_size = cfg.model.batch_size, shuffle=True, collate_fn=collate_fn)

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