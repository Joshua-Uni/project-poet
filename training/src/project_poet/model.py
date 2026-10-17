import torch
import torch.nn as nn
import torch.nn.functional as F

from project_poet.config import CFG


class FeedForward(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        out_proj = nn.Linear(4 * cfg.model.embedding_dim, cfg.model.embedding_dim)
        object.__setattr__(out_proj, "is_residual_proj", True)
        self.net = nn.Sequential(
            nn.Linear(cfg.model.embedding_dim, cfg.model.embedding_dim * 4),
            nn.GELU(),
            out_proj,
            nn.Dropout(cfg.model.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,C
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.n_head = cfg.model.n_head
        self.head_size = cfg.model.head_size
        # One linear for every head's query, key and value: [q heads | k heads | v heads].
        self.qkv = nn.Linear(cfg.model.embedding_dim, 3 * self.n_head * self.head_size)
        self.proj = nn.Linear(self.n_head * self.head_size, cfg.model.embedding_dim)
        object.__setattr__(self.proj, "is_residual_proj", True)
        self.attn_dropout = cfg.model.dropout
        self.dropout = nn.Dropout(cfg.model.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.n_head * self.head_size, dim=-1)  # 3 x B,T,nH*Hs
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # B,nH,T,Hs
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # B,nH,T,Hs
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # B,nH,T,Hs

        # Functional call, so dropout must be disabled manually outside training.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )  # B,nH,T,Hs
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)
        return self.dropout(self.proj(out))  # B,T,nH*Hs @ nH*Hs,C --> B,T,C


class Block(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.self_attention = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.model.embedding_dim)
        self.ln2 = nn.LayerNorm(cfg.model.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class PoetTransformer(nn.Module):
    def __init__(self, cfg: CFG, vocab_size: int):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, cfg.model.embedding_dim
        )  # shared with the output projection
        self.position_embedding = nn.Embedding(cfg.model.max_block_size, cfg.model.embedding_dim)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.model.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.model.embedding_dim)  # final layer norm

        self.n_layers = cfg.model.n_layers
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_proj", False):
                std = 0.02 / (2 * self.n_layers) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]

        tok_emb = self.token_embedding(idx)  # Each (B,T) indexes into (vocab index, C) -->(B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C) + (T,C) --> (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        # Weight is (vocab, C). F.linear computes x @ weight.T, so no transpose is stored.
        logits = F.linear(x, self.token_embedding.weight)  # (B,T,C) @ (vocab,C).T --> (B,T,vocab)

        return logits


# Reference: the explicit single-head attention this module replaces. Each head built and
# stored the full T x T weight matrix, and MultiHeadAttention ran the heads one after another
# and concatenated them. F.scaled_dot_product_attention computes the same result in one fused
# kernel that never materialises the T x T matrix.
#
# class Head(nn.Module):
#     def __init__(self, cfg: CFG):
#         super().__init__()
#         self.key = nn.Linear(cfg.model.embedding_dim, cfg.model.head_size)
#         self.query = nn.Linear(cfg.model.embedding_dim, cfg.model.head_size)
#         self.value = nn.Linear(cfg.model.embedding_dim, cfg.model.head_size)
#
#         self.tril: torch.Tensor
#         # Rebuilt in __init__ from max_block_size, so it is not saved in the checkpoint.
#         self.register_buffer(
#             "tril",
#             torch.tril(torch.ones(cfg.model.max_block_size, cfg.model.max_block_size)),
#             persistent=False,
#         )
#         self.dropout = nn.Dropout(cfg.model.dropout)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         T = x.shape[1]
#         k = self.key(x)  # B,T,Hs
#         q = self.query(x)  # B,T,Hs
#         v = self.value(x)  # B,T,Hs
#
#         weights = q @ k.transpose(-2, -1)  # B,T,Hs @ B,Hs,T --> B,T,T
#         weights = weights * k.shape[-1] ** -0.5  # Normalise Weights
#         weights = weights.masked_fill(
#             self.tril[:T, :T] == 0, float("-inf")
#         )  # -inf at future token's
#         weights = F.softmax(weights, dim=-1)  # 0 at future tokens
#         weights = self.dropout(weights)
#
#         return weights @ v  # B,T,T @ B,T,Hs --> B,T,Hs