"""Fused scaled_dot_product_attention against the explicit per-head formula."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from project_poet.config import Model as ModelCfg
from project_poet.model import MultiHeadAttention


def attention(dropout: float = 0.0) -> MultiHeadAttention:
    cfg = ModelCfg(
        embedding_dim=8, max_block_size=16, n_layers=1, n_head=2, head_size=4, dropout=dropout
    )
    return MultiHeadAttention(SimpleNamespace(model=cfg))


def explicit_attention(mha: MultiHeadAttention, x: torch.Tensor) -> torch.Tensor:
    # The old Head maths, run head by head on slices of the fused qkv weights.
    T = x.shape[1]
    width = mha.n_head * mha.head_size
    q_all, k_all, v_all = mha.qkv(x).split(width, dim=-1)
    tril = torch.tril(torch.ones(T, T))
    heads = []
    for h in range(mha.n_head):
        cols = slice(h * mha.head_size, (h + 1) * mha.head_size)
        q, k, v = q_all[..., cols], k_all[..., cols], v_all[..., cols]
        weights = q @ k.transpose(-2, -1) * mha.head_size**-0.5
        weights = weights.masked_fill(tril == 0, float("-inf"))
        heads.append(F.softmax(weights, dim=-1) @ v)
    return mha.proj(torch.cat(heads, dim=-1))


def test_fused_attention_matches_explicit_heads():
    torch.manual_seed(0)
    mha = attention().eval()
    x = torch.randn(3, 10, 8)

    torch.testing.assert_close(mha(x), explicit_attention(mha, x))


def test_future_tokens_do_not_change_earlier_outputs():
    torch.manual_seed(0)
    mha = attention().eval()
    x = torch.randn(1, 10, 8)
    changed = x.clone()
    changed[:, 6:] = torch.randn(1, 4, 8)

    torch.testing.assert_close(mha(x)[:, :6], mha(changed)[:, :6])


def test_eval_mode_disables_attention_dropout():
    torch.manual_seed(0)
    mha = attention(dropout=0.5).eval()
    x = torch.randn(2, 10, 8)

    torch.testing.assert_close(mha(x), mha(x))
