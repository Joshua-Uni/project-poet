"""Collate shift, deterministic split, and max_block_size slicing."""

from types import SimpleNamespace

import pandas as pd
import torch
from project_poet.config import Data as DataCfg
from project_poet.dataloader import build_sequences, make_collate_fn, split_dataframe


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
    #split_dataframe twice with the same seed and val_ratio returns the same index sets, and the val fraction matches.
    poem_df = pd.DataFrame({
        "Title": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "Genre": ["Love", "Joy", "Fear", "Anger", "Sadness", "Surprise", "Love", "Joy"],
        "Poem": ["one", "two", "three", "four", "five", "six", "seven", "eight"],
    })
    val_ratio = 0.2
    data_cfg = DataCfg("", 7899, val_ratio)
    cfg = SimpleNamespace(data=data_cfg)
    train1, val1 = split_dataframe(poem_df, cfg)
    train2, val2 = split_dataframe(poem_df, cfg)
    assert list(train1.index) == list(train2.index)
    assert list(val1.index) == list(val2.index)
    assert len(val1) == round(val_ratio * len(poem_df))


def test_block_size_slices_the_example(tokenizer):
    #build_sequences(..., max_block_size=4) returns lists of length at most 4, so a long poem is cut before the model sees it.
    max_block_size = 4
    long_poem = "Wow this is a very very very very long poem. I sure hope there's not max_block_size!!!"
    poem_df = pd.DataFrame({
        "Title": ["Title", "Title 2"],
        "Genre": ["Love", "Joy"],
        "Poem": ["Hello how are you!!", long_poem],
    })

    full = tokenizer.encode_example("Title 2", "Joy", long_poem)
    seq = build_sequences(poem_df, tokenizer, max_block_size)
    assert len(full) > max_block_size
    assert seq[1] == full[:max_block_size]
    assert all(len(item) <= max_block_size for item in seq)
