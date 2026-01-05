import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from convformer_lite.model import ConvFormerLite
from convformer_lite.utils import accuracy_top1


def main(cfg_path="configs/default.yaml", ckpt_path="checkpoints/best.pt"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = cfg["data"]["img_size"]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=tfm)
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = ConvFormerLite(
        num_classes=cfg["model"]["num_classes"],
        widths=tuple(cfg["model"]["widths"]),
        depths=tuple(cfg["model"]["depths"]),
        attn_every=int(cfg["model"]["attn_every"]),
        token_budget=int(cfg["model"]["token_budget"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    accs = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="eval"):
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            accs.append(accuracy_top1(logits, targets))

    print(f"Top-1 accuracy: {sum(accs)/len(accs):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/default.yaml")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()
    main(args.cfg, args.ckpt)
