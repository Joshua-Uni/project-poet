import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    opt: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device) -> float:
    model.train()
    total = 0.0
    n_tokens = 0
    for xb, yb, pad_mask in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        logits = model(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        opt.step()

        batch_tokens = int((yb != -100).sum())
        total += loss.item() * batch_tokens
        n_tokens += batch_tokens
        #print(loss.item())
    return total / n_tokens