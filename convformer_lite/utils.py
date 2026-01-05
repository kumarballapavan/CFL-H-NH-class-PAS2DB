import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_ckpt(save_dir, epoch, model, optimizer, best=False):
    os.makedirs(save_dir, exist_ok=True)
    name = "best.pt" if best else f"epoch_{epoch}.pt"
    path = os.path.join(save_dir, name)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path
