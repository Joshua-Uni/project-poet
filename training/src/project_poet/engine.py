import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from project_poet.config import Train as TrainCfg


def warmup_cosine(
    opt: torch.optim.Optimizer, train_cfg: TrainCfg, total_steps: int, start_step: int = 0
) -> LambdaLR:
    # Linear warmup to learning_rate, then cosine decay to learning_rate * min_lr_ratio.
    # Stepped once per batch. total_steps comes from the current config, so changing epochs on
    # resume reshapes the rest of the curve; start_step is where a resumed run picks it up.
    warmup = train_cfg.warmup_steps
    floor = train_cfg.min_lr_ratio

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        # Clamped so a run past total_steps (epochs lowered on resume) stays at the floor.
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    # The peak comes from the config, not the optimizer, whose lr is mid-schedule on resume.
    for group in opt.param_groups:
        group["initial_lr"] = train_cfg.learning_rate
    return LambdaLR(opt, factor, last_epoch=start_step - 1)


def train_one_epoch(
    model: nn.Module,
    train_cfg: TrainCfg,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scheduler: LambdaLR,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n_tokens = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        logits = model(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_norm)
        opt.step()
        scheduler.step()

        batch_tokens = int((yb != -100).sum())
        total += loss.item() * batch_tokens
        n_tokens += batch_tokens
    return total / n_tokens


def eval_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    model.eval()
    n_tokens = 0
    total = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))

            batch_tokens = int((yb != -100).sum())
            total += loss.item() * batch_tokens
            n_tokens += batch_tokens

    return total / n_tokens


def generate_text(
    model: nn.Module,
    text: list[int],
    max_block_size: int,
    temperature: float,
    device: torch.device,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    model.eval()

    with torch.no_grad():
        x = torch.tensor(text, dtype=torch.long).to(device)
        x = x[-max_block_size:]
        x = x.unsqueeze(0)
        logits = model(x)
        # last-token logits. temperature, then top-k, then top-p, then one softmax
        logits = logits[0, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            # keep the token that crosses top_p; drop the ones after it
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_logits = sorted_logits.masked_fill(
                cumulative - sorted_probs > top_p, float("-inf")
            )
            logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)

    return int(torch.multinomial(probs, 1).item())
