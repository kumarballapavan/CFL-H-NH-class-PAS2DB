# CFL Net (PyTorch)

## Install
```bash
pip install -r requirements.txt
```

## Data format
Uses `torchvision.datasets.ImageFolder`:

```
data/
  train/
    class0/xxx.png
    class1/yyy.png
  val/
    class0/...
    class1/...
```

## Train
```bash
python train.py --cfg configs/default.yaml
```

## Evaluate
```bash
python eval.py --cfg configs/default.yaml --ckpt checkpoints/best.pt
```

## Notes
- Edit `configs/default.yaml` for widths/depths/token budget.
- `token_budget` keeps attention compute bounded.


## Dataset
Sample data (pre-publication): We provide a minimal sample subset containing the 6 hazy/clear paired examples used in the manuscript to enable quick verification.
The details of the complete dataset will be shared post publication.
