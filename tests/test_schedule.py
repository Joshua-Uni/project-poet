"""Warmup-then-cosine learning rate schedule, including resume."""

import pytest
import torch
from project_poet.config import Train
from project_poet.engine import warmup_cosine

LR = 1e-3


def train_cfg(warmup_steps: int = 10, min_lr_ratio: float = 0.1) -> Train:
    return Train(
        resume=False,
        model_path="",
        best_path="",
        log_lvl="INFO",
        learning_rate=LR,
        weight_decay=0.0,
        max_norm=1.0,
        epochs=1,
        batch_size=1,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )


def make_opt(lr: float = LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def lrs(
    cfg: Train, total_steps: int, n: int, start_step: int = 0, opt_lr: float = LR
) -> list[float]:
    # The lr used for each of the next n batches.
    opt = make_opt(opt_lr)
    scheduler = warmup_cosine(opt, cfg, total_steps, start_step)
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
    return out


def test_warms_up_to_the_peak_then_decays_to_the_floor():
    # 10 warmup steps of 100: step 0 is lr/10, step 9 is the peak, the last step is near lr * 0.1.
    seq = lrs(train_cfg(), total_steps=100, n=100)
    assert seq[0] == pytest.approx(LR / 10)
    assert seq[9] == pytest.approx(LR)
    assert all(a >= b for a, b in zip(seq[10:], seq[11:], strict=False))
    assert seq[-1] == pytest.approx(LR * 0.1, rel=1e-2)  # the last step is 89/90 of the way down


def test_stays_at_the_floor_past_total_steps():
    # Lowering epochs on resume can put the run past total_steps; the lr must not climb back up.
    seq = lrs(train_cfg(), total_steps=50, n=80)
    assert seq[60:] == pytest.approx([LR * 0.1] * 20)


def test_resume_picks_up_the_same_curve_whatever_the_optimizer_lr():
    # A resumed optimizer carries a mid-schedule lr; the schedule restarts from the config peak.
    full = lrs(train_cfg(), total_steps=100, n=100)
    resumed = lrs(train_cfg(), total_steps=100, n=40, start_step=60, opt_lr=1e-5)
    assert resumed == pytest.approx(full[60:])


def test_no_warmup_and_a_floor_of_one_is_a_constant_learning_rate():
    assert lrs(train_cfg(warmup_steps=0, min_lr_ratio=1.0), total_steps=20, n=20) == pytest.approx(
        [LR] * 20
    )
