"""Forward shape and the tied token embedding."""

import torch
from project_poet.model import PoetTransformer


def test_forward_logits_match_batch_time_vocab(tiny_model: PoetTransformer):
    #A tiny PoetTransformer given idx of shape (2, 5) returns logits of shape (2, 5, vocab).
    #vocab = is 32
    data = torch.randint(0, 32, (2,5))
    logits = tiny_model(data)
    B,T,vocab = logits.shape

    assert (B,T,vocab)== (2,5,32)


def test_output_projection_reuses_the_embedding(tiny_model: PoetTransformer):
    weight = tiny_model.token_embedding.weight
    vocab = weight.size(0)

    assert not any(name.startswith("lm_head") for name in tiny_model.state_dict())
    vocab_tables = [
        name
        for name, module in tiny_model.named_modules()
        if isinstance(module, torch.nn.Embedding) and module.num_embeddings == vocab
    ]
    assert vocab_tables == ["token_embedding"]

    idx = torch.randint(0, vocab, (2, 5))
    logits = tiny_model(idx)
    assert logits.shape[-1] == vocab
    logits.sum().backward()
    assert weight.grad is not None