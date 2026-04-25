# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Added

- `evaluate.py` — Full evaluation module for post-training analysis.
  - Top-1 / Top-5 Accuracy using `sklearn.metrics.top_k_accuracy_score`
  - Per-class Accuracy across all 100 CIFAR-100 categories
  - Confusion Matrix heatmap (Top-20 most confused classes)
  - Classification Report with Precision, Recall, and F1-score per class, exported as `classification_report.txt`
  - Loss / Accuracy curve plots generated from `training_log.json`
  - Summary exported as `summary.json` including overall metrics and best/worst 5 classes
  - Usage: `python evaluate.py --checkpoint ./checkpoints/model_best.pth`

- `colab_train.ipynb` — Google Colab training notebook.
  - Automatically mounts Google Drive for persistent checkpoint storage
  - Clones or pulls latest code from GitHub on each session
  - Detects existing checkpoints and resumes training automatically
  - Designed for GPU-accelerated training on Colab T4 / P100

### Changed

- `train.py` — Added training log export.
  - Added `import json`
  - Reads existing `training_log.json` on startup to preserve history when resuming
  - Appends `train_loss`, `train_acc`, `val_loss`, `val_acc` to `training_log.json` after each epoch
  - Log file is saved to the directory specified by `--save-dir`

- `requirements.txt` — Added `scikit-learn>=1.0` for evaluation metrics.

- `README.md` — Updated to reflect new files and evaluation workflow.

---

## [0.1.0] — Initial Release

### Added

- `train.py` — Training and validation loop with tqdm progress bar, SGD optimizer, StepLR scheduler, and automatic checkpoint saving.
- `model.py` — ResNet-18 adapted for 32x32 CIFAR-100 images (100 output classes).
- `utils.py` — `AverageMeter`, `accuracy` (Top-1), and `save_checkpoint` helpers.
- `requirements.txt` — Core dependencies: `torch`, `torchvision`, `tqdm`, `matplotlib`.
- `README.md` — Project setup, training, and resume instructions.
