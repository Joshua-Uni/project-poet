"""Collate shift, deterministic split, and max_block_size slicing."""

import torch
import pandas as pd

from project_poet.dataloader import build_sequences, make_collate_fn
from project_poet.tokenizer import PAD_TOKEN


def test_collate_shifts_targets_and_ignores_pad(tokenizer):
    #Two lengths. x is seq[:-1] padded with pad_id. y is seq[1:] padded with -100.
    seq1 = torch.tensor([19, 100, 100, 200])
    seq2 = torch.tensor([10, 23, 30])

    collate_fn = make_collate_fn(tokenizer.pad_id)

    x,y = collate_fn([seq1, seq2])
    x2, y2 = x[1], y[1]
    #Check padding on x and y
    assert x2[-1] == tokenizer.pad_id
    assert y2[-1] == -100

    #Check targets shift
    assert x2[1] == y2[0]



def test_split_is_deterministic():
    pass
    

def test_block_size_slices_the_example(tokenizer):
    #build_sequences(..., max_block_size=4) returns lists of length at most 4, so a long poem is cut before the model sees it.
    max_block_size = 4
    poem_df = pd.DataFrame({
        "Title": ["Title", "Title 2"],
        "Genre": ["Love", "Joy"],
        "Poem": ["Hello how are you!!", "Wow this is a very very very very long poem. I sure hope there's not max_block_size!!!"],
        })

    seq = build_sequences(poem_df, tokenizer, max_block_size)
    for s in seq:
        assert len(s) <= max_block_size
