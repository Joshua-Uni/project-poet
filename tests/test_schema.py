"""GenerateRequest defaults, genre, title, and sampling bounds."""

import pytest
from poet_api.schemas import GenerateRequest


def test_request_defaults_leave_sampling_open():
    # GenerateRequest(title='Night') has genre Love, temperature 1, max_tokens 400, top_k None, top_p None.
    req = GenerateRequest(title="Night")
    assert req.genre == "Love"
    assert req.temperature == 1
    assert req.max_tokens == 400
    assert req.top_k is None
    assert req.top_p is None


def test_request_rejects_titles_longer_than_80_characters():
    # 80 characters is the schema cap, so the prompt cannot grow past the block size.
    assert GenerateRequest(title="a" * 80).title == "a" * 80
    with pytest.raises(ValueError):
        GenerateRequest(title="a" * 81)


def test_request_rejects_blank_title_and_unknown_genre():
    # Title ' ' and genre 'Nature' raise ValidationError. 'Love' is accepted.
    with pytest.raises(ValueError):
        GenerateRequest(title=" ")

    with pytest.raises(ValueError):
        GenerateRequest(genre="Nature")
    assert GenerateRequest(title="Night", genre="Love").genre == "Love"


def test_request_bounds_on_sampling_knobs():
    # top_k 0 and 10000 fail; 1 and 9999 pass. top_p 0 fails; 1 passes. temperature 0 fails. max_tokens 0 and 701 fail; 700 passes.
    # Omitted top_k and top_p stay None.
    with pytest.raises(ValueError):
        GenerateRequest(top_k=0)
    with pytest.raises(ValueError):
        GenerateRequest(top_k=10_000)
    assert GenerateRequest(top_k=1).top_k == 1
    assert GenerateRequest(top_k=9999).top_k == 9999

    with pytest.raises(ValueError):
        GenerateRequest(top_p=0)
    assert GenerateRequest(top_p=1).top_p == 1

    with pytest.raises(ValueError):
        GenerateRequest(temperature=0)

    with pytest.raises(ValueError):
        GenerateRequest(max_tokens=0)
    with pytest.raises(ValueError):
        GenerateRequest(max_tokens=701)
    assert GenerateRequest(max_tokens=700).max_tokens == 700

    omitted = GenerateRequest(title="Night")
    assert omitted.top_k is None
    assert omitted.top_p is None
