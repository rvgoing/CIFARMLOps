# CIFAR-100 PyTorch Training Project

This project provides a full training pipeline for CIFAR-100 using PyTorch.

## Files

- `train.py`: Training and validation loop.
- `model.py`: ResNet-18 model adapted for 32x32 CIFAR images.
- `utils.py`: Helpers for checkpointing and accuracy measurement.
- `requirements.txt`: Required Python packages.

## Setup

1. Create a Python environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run training

```bash
python train.py --data-dir ./data --epochs 50 --batch-size 128 --lr 0.1
```

The script downloads CIFAR-100 automatically and saves checkpoints to `./checkpoints`.

## Resume training

```bash
python train.py --resume ./checkpoints/checkpoint.pth
```

## Notes

- The default model is ResNet-18 with CIFAR-100 output classes.
- The training uses standard CIFAR-100 normalization and data augmentation.
