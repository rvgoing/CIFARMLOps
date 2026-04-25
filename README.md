# CIFAR-100 PyTorch Training Project

This project provides a full training and evaluation pipeline for CIFAR-100 using PyTorch.

## Files

- `train.py`: Training and validation loop with epoch-level log export.
- `model.py`: ResNet-18 model adapted for 32x32 CIFAR images.
- `utils.py`: Helpers for checkpointing and accuracy measurement.
- `evaluate.py`: Full evaluation module — Top-1/5 Accuracy, Confusion Matrix, Per-class Accuracy, Classification Report, and Loss/Accuracy curves.
- `requirements.txt`: Required Python packages.
- `colab_train.ipynb`: Google Colab notebook for GPU training with automatic checkpoint resume from Google Drive.

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
A `training_log.json` is also saved to the same directory after each epoch.

## Resume training

```bash
python train.py --resume ./checkpoints/checkpoint.pth
```

Training resumes from the last saved epoch. The existing `training_log.json` is preserved and appended.

## Run evaluation

After training, run the evaluation module against the best checkpoint:

```bash
python evaluate.py --checkpoint ./checkpoints/model_best.pth
```

Results are saved to `./eval_results/` by default:

| File | Description |
|---|---|
| `summary.json` | Top-1 / Top-5 Accuracy, best and worst 5 classes |
| `classification_report.txt` | Per-class Precision, Recall, F1-score |
| `confusion_matrix.png` | Heatmap of the 20 most confused classes |
| `per_class_accuracy.png` | Bar chart of the 20 lowest-accuracy classes |
| `loss_acc_curve.png` | Training and validation Loss / Accuracy curves |

## Google Colab Training

Open `colab_train.ipynb` in Google Colab for GPU-accelerated training.  
The notebook automatically mounts Google Drive, pulls the latest code from GitHub, detects existing checkpoints, and resumes training if available.

## Notes

- The default model is ResNet-18 with CIFAR-100 output classes.
- The training uses standard CIFAR-100 normalization and data augmentation.
- Evaluation requires `scikit-learn` and `matplotlib` (included in `requirements.txt`).