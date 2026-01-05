import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from convformer_lite.model import ConvFormerLite
from convformer_lite.utils import set_seed, accuracy_top1, save_ckpt


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path="configs/default.yaml"):
    cfg = load_cfg(cfg_path)

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = cfg["data"]["img_size"]

    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=tfm_train)
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=tfm_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = ConvFormerLite(
        num_classes=cfg["model"]["num_classes"],
        widths=tuple(cfg["model"]["widths"]),
        depths=tuple(cfg["model"]["depths"]),
        attn_every=int(cfg["model"]["attn_every"]),
        token_budget=int(cfg["model"]["token_budget"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)))

    best_acc = -1.0
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg["train"].get("amp", True))):
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        # val
        model.eval()
        accs = []
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                accs.append(accuracy_top1(logits, targets))

        val_acc = sum(accs) / max(1, len(accs))
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}")

        save_ckpt(save_dir, epoch, model, optimizer, best=False)
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(save_dir, epoch, model, optimizer, best=True)
            print(f"✅ New best: {best_acc:.4f}")

    print("Done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    main(args.cfg)
