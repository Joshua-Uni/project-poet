import torch
import torch.nn as nn
import torch.nn.functional as F

from project_poet.config import CFG

### DONT FORGET TO PUT THESE IN CONFIG
N_EMBD = 16
DROPOUT = 0.2
N_HEAD = 2
HEAD_SIZE = 8
MAX_BLOCK_SIZE = 9000
N_LAYERS = 2



class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, N_EMBD*4), #B,T,C*4
            nn.ReLU(),
            nn.Linear(4* N_EMBD, N_EMBD), #B,T,C
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, x):
        #x: B,T,C
        return self.net(x)



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()    
        self.key = nn.Linear(N_EMBD, head_size)
        self.query = nn.Linear(N_EMBD, head_size)
        self.value = nn.Linear(N_EMBD, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE))) #Below Spine == 1 else 0
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #B,T,Hs
        q = self.query(x) #B,T,Hs
        v = self.value(x) #B,T,Hs

        weights = q @ k.transpose(-2, -1) #B,T,Hs @ B,Hs,T --> B,T,T
        weights = weights*k.shape[-1]**-0.5 #Normalise Weights
        weights = weights.masked_fill(self.tril[:T, :T]==0, float("-inf")) #-inf at future token's
        weights = F.softmax(weights, dim=-1) #0 at future tokens
        weights = self.dropout(weights)

        return weights @ v #B,T,T @ B,T,Hs --> B,T,Hs



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(HEAD_SIZE) for _ in range(N_HEAD)])
        self.proj = nn.Linear(HEAD_SIZE*N_HEAD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #N_HEAD {B,T,Hs} concats across HS --> B,T,N_HEAD*HS
        out = self.dropout(self.proj(out)) #B,T,N_HEAD*Hs @ N_HEAD*Hs, C --> B,T,C
        return out



class Block(nn.Module):
    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class PoetTransformer(nn.Module):
    def __init__(self, cfg: CFG, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, cfg.model.embedding_dim)
        self.position_embedding = nn.Embedding(MAX_BLOCK_SIZE, cfg.model.embedding_dim)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(cfg.model.embedding_dim) # final layer norm
        self.lm_head = nn.Linear(cfg.model.embedding_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding(idx) # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits