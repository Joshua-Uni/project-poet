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



def eval_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   loss_fn: nn.Module,
                   device: torch.device) -> float:
    model.eval()
    n_tokens = 0
    total = 0

    with torch.no_grad():
        for xb, yb, pad_mask in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))

            batch_tokens = int((yb != -100).sum())
            total += loss.item() * batch_tokens
            n_tokens += batch_tokens
        
    return total / n_tokens


 
def generate_text(model: nn.Module, text: list, max_block_size: int, temperature: float, device: torch.device):
    model.eval()

    with torch.no_grad():
        x = torch.tensor(text, dtype=torch.long).to(device)
        x = x[-max_block_size:]
        x = x.unsqueeze(0)
        #print(x.shape)
        logits = model(x)
        #print(logits.shape)
        probs = torch.softmax(logits[0,-1,:] / temperature, dim=-1)

    return torch.multinomial(probs, 1).item()

