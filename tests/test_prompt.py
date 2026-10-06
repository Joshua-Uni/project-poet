"""Prompt order, encode_example, genre ids, and BPE training text."""

import pandas as pd
from project_poet.tokenizer import GENRES, PoemTokenizer, _train_texts


def test_prompt_puts_genre_after_poem_start(tokenizer: PoemTokenizer):
    genre = "Love"
    title = "Hello"
    prompt_id = tokenizer.prompt_ids(title, genre)

    assert prompt_id[0] == tokenizer.title_start_id
    assert prompt_id[-3] == tokenizer.title_end_id
    assert prompt_id[-2] == tokenizer.poem_start_id
    assert prompt_id[-1] == tokenizer.genre_id[genre]
    assert tokenizer.poem_end_id not in prompt_id


def test_encode_example_extends_the_prompt(tokenizer: PoemTokenizer):
    genre = "Love"
    title = "Hello"
    poem = "Hello World, how are you?"
    assert tokenizer.encode_example(title, genre, poem) == tokenizer.prompt_ids(title, genre) + tokenizer.encode(poem) + [tokenizer.poem_end_id]


def test_genre_tokens_are_registered(tokenizer: PoemTokenizer):
    ids = [tokenizer.genre_id[name] for name in GENRES]
    assert None not in ids
    assert len(set(ids)) == len(GENRES)


def test_bpe_sees_title_and_poem_only():
    poem_df = pd.DataFrame({
        "Title": ["Title", "Title 2"],
        "Poem": ["Poem", "Poem 2"],
        "Genre": ["Love", "Fear"],
        "Love": [0.9, 0.1],
    })
    assert list(_train_texts(poem_df)) == ["Title", "Poem", "Title 2", "Poem 2"]