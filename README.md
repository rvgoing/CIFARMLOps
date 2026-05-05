# CIFARMLOps — CIFAR-100 MLOps Training System

> A production-oriented MLOps pipeline built on CIFAR-100, demonstrating end-to-end model training, experiment tracking, evaluation, and deployment readiness.

---

## 🎯 Project Goal

This project is **not** a simple training tutorial.

It is a progressively evolving MLOps system that demonstrates:

- ✅ Reproducible training pipeline with automatic Checkpoint / Resume
- ✅ Experiment tracking with MLflow
- ✅ Comprehensive model evaluation (Top-1/5, Confusion Matrix, Per-class Accuracy, F1)
- ✅ Cloud GPU training via Google Colab with Google Drive persistence
- ✅ Version-controlled codebase (GitHub)
- 🔜 Model deployment (FastAPI + Docker)
- 🔜 CI/CD automation (GitHub Actions)
- 🔜 Online monitoring (Model Drift Detection)

---

## 🗂️ Project Structure

```
CIFARMLOps/
├── train.py              # Training pipeline with MLflow tracking
├── evaluate.py           # Full evaluation module
├── model.py              # ResNet-18 adapted for CIFAR-100
├── utils.py              # Checkpoint / Accuracy helpers
├── demo.py               # Gradio interactive demo
├── requirements.txt      # Python dependencies
├── colab_train.ipynb     # Colab GPU training notebook
├── CHANGELOG.md          # Version history
└── README.md             # This file
```

---

## 🚀 MLOps Architecture

```
Local VS Code (develop)
        ↓
GitHub (version control)
        ↓
Google Colab (GPU training)
        ↓
MLflow (experiment tracking)
        ↓
Google Drive (checkpoint persistence)
        ↓
evaluate.py (model evaluation)
        ↓
demo.py (interactive demo)
```

---

## ⚙️ Setup

### Local Environment

```bash
git clone https://github.com/rvgoing/CIFARMLOps.git
cd CIFARMLOps
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

---

## 🏋️ Training

### Local Training

```bash
python train.py --epochs 200 --batch-size 128 --lr 0.1
```

### Resume Training

```bash
python train.py --epochs 200 --resume ./checkpoints/checkpoint.pth
```

### Training with MLflow Tracking

```bash
python train.py \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.1 \
    --mlflow-dir ./mlruns \
    --exp-name CIFAR100_Exp01
```

CIFAR-100 dataset is downloaded automatically on first run.  
Checkpoints and `training_log.json` are saved to `--save-dir` after each epoch.

---

## ☁️ Google Colab Training

Open `colab_train.ipynb` in Google Colab for free GPU training.

The notebook handles everything automatically:

| Step | Action |
|---|---|
| Step 1 | Mount Google Drive |
| Step 2 | Clone / Pull latest code from GitHub |
| Step 3 | Install dependencies |
| Step 4 | Verify GPU |
| Step 5 | Auto-detect checkpoint → Resume or start fresh |
| Step 6 | Verify checkpoint saved to Drive |
| Step 7 | Run evaluation after training |
| Step 8 | Clear checkpoints (run only when needed) |

---

## 📊 MLflow Experiment Tracking

Every training run automatically records:

| Category | Details |
|---|---|
| Parameters | lr, epochs, batch_size, optimizer, scheduler, model, dataset |
| Metrics (per epoch) | train_loss, train_acc, val_loss, val_acc |
| Final metrics | best_val_acc, best_epoch |
| Artifacts | model_best.pth |

### View MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://localhost:5000` to compare experiments visually.

---

## 📈 Evaluation

Run the evaluation module after training:

```bash
python evaluate.py --checkpoint ./checkpoints/model_best.pth
```

Outputs saved to `./eval_results/`:

| File | Description |
|---|---|
| `summary.json` | Top-1 / Top-5 Accuracy, best and worst 5 classes |
| `classification_report.txt` | Per-class Precision, Recall, F1-score |
| `confusion_matrix.png` | Heatmap of 20 most confused classes |
| `per_class_accuracy.png` | Bar chart of 20 lowest-accuracy classes |
| `loss_acc_curve.png` | Training and validation Loss / Accuracy curves |
| `labels.npy` / `preds.npy` | Raw inference results for further analysis |

---

## 🧪 Experiment History

| Exp | epochs | lr | Augmentation | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|---|
| Exp-01 | 20 | 0.1 | Basic | 56.45% | 84.68% | Baseline, undertrained |
| Exp-02 | 200 | 0.1 | Basic | 77.74% | 94.16% | Overfitting observed |
| Exp-03 | - | - | - | - | - | Planned: add stronger augmentation |

---

## 🎮 Interactive Demo

```bash
pip install gradio
python demo.py --checkpoint ./checkpoints/model_best.pth
```

For Colab (generates public URL):

```bash
python demo.py --checkpoint ./checkpoints/model_best.pth --share
```

---

## 🗺️ MLOps Roadmap

```
Phase 1 ✅  Training Pipeline
            - Reproducible training with Checkpoint / Resume
            - Automatic training log export

Phase 2 ✅  Experiment Tracking
            - MLflow integration
            - Per-epoch metrics logging
            - Model artifact management

Phase 3 🔜  Model Deployment
            - FastAPI REST API
            - Docker containerization

Phase 4 🔜  CI/CD Automation
            - GitHub Actions triggered training
            - Automated evaluation on push

Phase 5 🔜  Online Monitoring
            - Data Drift detection
            - Model Drift detection
            - Auto retraining trigger
```

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
tqdm
matplotlib>=3.0
scikit-learn>=1.0
mlflow>=2.0
gradio
```

---

## 📝 Notes

- Default model is ResNet-18 adapted for 32×32 CIFAR-100 images (100 classes)
- Standard CIFAR-100 normalization and data augmentation applied
- Overfitting observed after 200 epochs (Train Acc ~99%, Val Acc ~77%) — next experiments will address this
