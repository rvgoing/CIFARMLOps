# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Planned
- FastAPI model deployment
- Docker containerization
- GitHub Actions CI/CD
- Online monitoring (Data Drift / Model Drift)

---

## [0.4.0] — Training Correctness & Reproducibility

### Fixed
- `train.py` — `best_epoch` now records the epoch with the highest validation accuracy (it logged the last epoch)
- `train.py` — Validation uses a 5k split of the training set (`--val-size`); the test set is no longer used for model selection
- `train.py` — Short runs now decay the learning rate: default schedule is CosineAnnealingLR over `--epochs` (`--scheduler step` keeps StepLR)
- `train.py` — Resume continues the same MLflow run instead of opening a new one
- `train.py` — `training_log.json` is reset on fresh runs and trimmed to the checkpoint epoch on resume
- `train.py` — Checkpoint is saved after `scheduler.step()`, so a resumed run no longer repeats one epoch at the previous learning rate

### Added
- `train.py` — `--seed` (default 42) and `--deterministic`; per-epoch reseeding makes crash + resume match an uninterrupted run
- `train.py` — `lr` logged per epoch; `seed` and `val_size` logged as params

### Note
- Validation accuracy from 0.4.0 runs is measured on the held-out split, so it is not directly comparable to earlier `best_val_acc` values. Use `evaluate.py` (test set) to compare runs.

---

## [0.3.0] — MLflow Integration

### Added
- `train.py` — MLflow experiment tracking
  - `--mlflow-dir` argument for tracking storage path
  - `--exp-name` argument for experiment name
  - `mlflow.log_params()` — records all hyperparameters at run start
  - `mlflow.log_metrics()` — records train/val loss and accuracy per epoch
  - `mlflow.log_artifact()` — saves best model checkpoint to MLflow
  - Prints MLflow Run ID on training completion

- `requirements.txt` — Added `mlflow>=2.0`

### Changed
- `colab_train.ipynb` — Updated Step 3 to show per-package install status
- `colab_train.ipynb` — Added `--mlflow-dir` parameter to training command
- `colab_train.ipynb` — Renamed Step 8 (Clear Checkpoints) for clarity
- `README.md` — Restructured to reflect MLOps positioning

---

## [0.2.0] — Evaluation System

### Added
- `evaluate.py` — Full post-training evaluation module
  - Top-1 / Top-5 Accuracy
  - Per-class Accuracy across all 100 CIFAR-100 categories
  - Confusion Matrix heatmap (Top-20 most confused classes)
  - Classification Report (Precision / Recall / F1-score)
  - Loss / Accuracy curve plots from `training_log.json`
  - Summary JSON with overall metrics and best/worst 5 classes
  - Saves `labels.npy` and `preds.npy` for downstream analysis

- `demo.py` — Gradio interactive demo
  - Upload any image for Top-5 classification
  - Random sample button with ground truth display

- `colab_train.ipynb` — Added Step 7 (Evaluation) and Step 8 (Clear Checkpoints)
- `CHANGELOG.md` — Added version history

### Changed
- `train.py` — Added `training_log.json` export per epoch
- `requirements.txt` — Added `scikit-learn>=1.0`
- `README.md` — Updated to document evaluation workflow

---

## [0.1.0] — Initial Release

### Added
- `train.py` — Training and validation loop with tqdm progress bar, SGD optimizer, StepLR scheduler, and automatic checkpoint saving
- `model.py` — ResNet-18 adapted for 32×32 CIFAR-100 images (100 output classes)
- `utils.py` — `AverageMeter`, `accuracy` (Top-1), and `save_checkpoint` helpers
- `colab_train.ipynb` — Google Colab training notebook with automatic checkpoint resume from Google Drive
- `requirements.txt` — Core dependencies
- `README.md` — Project setup and training instructions

---

## Experiment Log

| Exp | epochs | lr | Augmentation | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|---|
| Exp-01 | 20 | 0.1 | Basic | 56.45% | 84.68% | Baseline, undertrained |
| Exp-02 | 200 | 0.1 | Basic | 77.74% | 94.16% | Overfitting observed |
| Exp-03 | - | - | - | - | - | Planned |