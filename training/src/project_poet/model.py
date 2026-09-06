import torch
import torch.nn as nn

from project_poet.config import CFG


class PoetTransformer(nn.Module):
    def __init__(self, cfg: CFG, vocab_size: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, cfg.model.embedding_dim)
        #Dont forget time embedding eventually
        self.final_fc = nn.Linear(cfg.model.embedding_dim, vocab_size)

    
    def forward(self, x):
        x_emb = self.embedding(x)
        return self.final_fc(x_emb)