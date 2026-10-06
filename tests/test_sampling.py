"""top-k, top-p, and the vocab clamp in generate_text."""
import torch
import torch.nn as nn
from project_poet.engine import generate_text


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch, time = idx.shape
        return self.logits.view(1, 1, -1).expand(batch, time, -1)


def test_top_k_one_returns_the_argmax():
    logits = torch.tensor([2.0, 1.0, -1.0, -2.0])
    model = FixedLogits(logits)
    text = [0]
    device = torch.device("cpu")
    max_block_size = 10

    token = generate_text(model, text, max_block_size, 1.0, device, top_k=1)
    assert token == torch.argmax(logits).item()
    low_temp = generate_text(model, text, max_block_size, 0.2, device, top_k=1)
    assert low_temp == torch.argmax(logits).item()


def test_top_k_keeps_ties_at_the_cutoff():
    #Logits [2, 2, 1.9] and top_k=1. Forty draws all land in {0, 1}, because the mask drops logits strictly below the cutoff.
    logits = torch.tensor([2.0, 2.0, 1.9])
    model = FixedLogits(logits)
    text = [0]
    device = torch.device("cpu")
    max_block_size = 10

    tokens = [
        generate_text(model, text, max_block_size, 1.0, device, top_k=1)
        for _ in range(40)
    ]
    assert all(token in (0, 1) for token in tokens)


def test_top_p_drops_tokens_past_the_nucleus():
    #A peaked distribution and a small top_p return only the top token. top_p=1 still returns an in-range id. top_k=None is allowed.
    logits = torch.tensor([3.0, 2.0, 1.0])
    model = FixedLogits(logits)
    text = [0]
    device = torch.device("cpu")
    max_block_size = 10

    token = generate_text(model, text, max_block_size, 1.0, device, top_p=0.1)
    assert token == 0
    top_p_one = generate_text(model, text, max_block_size, 1.0, device, top_p=1)
    assert top_p_one in (0, 1, 2)


def test_top_k_larger_than_vocab_is_clamped():
    #top_k=10000 on a 4-way vocab does not raise. The sample is in 0..3. This is the clamp inside generate_text.
    logits = torch.tensor([3.0, 2.0, 1.0, 3.0])
    model = FixedLogits(logits)
    text = [0]
    device = torch.device("cpu")
    max_block_size = 10

    token = generate_text(model, text, max_block_size, 1.0, device, top_k=10000)
    assert token in (0, 1, 2, 3)