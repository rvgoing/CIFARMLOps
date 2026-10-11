import argparse
import json
import os
import random
import warnings

import numpy as np
from tqdm import tqdm

# Suppress NumPy 2.4+ deprecation warning from torchvision CIFAR loader
warnings.filterwarnings('ignore', message='.*dtype.*align should be passed as Python or NumPy boolean.*')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch
from mlflow.exceptions import MlflowException

from model import get_model
from utils import accuracy, save_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 training with PyTorch")
    parser.add_argument("--data-dir",     default="./data",        type=str,   help="Path to CIFAR-100 dataset folder")
    parser.add_argument("--epochs",       default=20,              type=int,   help="Number of training epochs")
    parser.add_argument("--batch-size",   default=128,             type=int,   help="Batch size for training and validation")
    parser.add_argument("--lr",           default=0.1,             type=float, help="Initial learning rate")
    parser.add_argument("--momentum",     default=0.9,             type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4,            type=float, help="Weight decay")
    parser.add_argument("--num-workers",  default=4,               type=int,   help="Data loader worker count")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to train on")
    parser.add_argument("--save-dir",     default="./checkpoints", type=str,   help="Directory to save models and logs")
    parser.add_argument("--resume",       default="",              type=str,   help="Path to checkpoint to resume from")
    # ── 可重現性 / 驗證集 / LR 排程 ──────────────────────────────────────────
    parser.add_argument("--seed",         default=42,              type=int,   help="Random seed (also fixes the train/val split)")
    parser.add_argument("--deterministic", action="store_true",               help="Use deterministic cuDNN kernels (slower, bit-exact on GPU)")
    parser.add_argument("--val-size",     default=5000,            type=int,   help="Images held out from the training set for validation / model selection")
    parser.add_argument("--scheduler",    default="cosine",        choices=["cosine", "step"], help="LR schedule: cosine over --epochs, or StepLR")
    parser.add_argument("--step-size",    default=60,              type=int,   help="StepLR step size (only with --scheduler step)")
    parser.add_argument("--gamma",        default=0.1,             type=float, help="StepLR decay factor (only with --scheduler step)")
    # ── MLflow 設定 ────────────────────────────────────────────────────────
    # 本地模式：  --mlflow-dir ./mlruns
    # Server模式：--mlflow-server http://your-server:5000
    parser.add_argument("--mlflow-dir",    default="./mlruns",  type=str, help="MLflow local tracking directory (used when --mlflow-server is not set)")
    parser.add_argument("--mlflow-server", default="",          type=str, help="MLflow tracking server URI (e.g. http://localhost:5000)")
    parser.add_argument("--exp-name",      default="CIFAR100",  type=str, help="MLflow experiment name")
    return parser.parse_args()


def setup_mlflow(args):
    """設定 MLflow Tracking URI：優先使用 Server，否則使用本地目錄"""
    if args.mlflow_server:
        mlflow.set_tracking_uri(args.mlflow_server)
        print(f"MLflow Tracking Server: {args.mlflow_server}")
    else:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(args.mlflow_dir)}")
        print(f"MLflow Local Dir: {os.path.abspath(args.mlflow_dir)}")
    mlflow.set_experiment(args.exp_name)


def set_seed(seed, deterministic=False):
    """固定 Python / NumPy / PyTorch 的亂數種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def build_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


def start_mlflow_run(run_id):
    """Resume 時接續同一個 MLflow run；找不到時（例如換了 tracking store）才開新的 run"""
    if run_id:
        try:
            run = mlflow.start_run(run_id=run_id)
            print(f"Continuing MLflow run {run_id}")
            return run, True
        except MlflowException as e:
            print(f"  ⚠️  Could not reopen MLflow run {run_id} ({e}); starting a new run")
    return mlflow.start_run(), False


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    set_seed(args.seed, args.deterministic)

    # ── MLflow 設定 ───────────────────────────────────────────────
    setup_mlflow(args)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # ── 從訓練集切出驗證集（測試集只留給 evaluate.py）─────────────────
    # 同一份訓練影像載入兩次：訓練部分用資料增強，驗證部分用 test transform
    train_full = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    val_full   = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_test)
    if not 0 < args.val_size < len(train_full):
        raise ValueError(f"--val-size must be between 1 and {len(train_full) - 1}, got {args.val_size}")
    split_gen     = torch.Generator().manual_seed(args.seed)  # 固定切分，Resume 時驗證集不變
    perm          = torch.randperm(len(train_full), generator=split_gen).tolist()
    train_dataset = Subset(train_full, perm[args.val_size:])
    val_dataset   = Subset(val_full,   perm[:args.val_size])
    print(f"Train / Val split: {len(train_dataset)} / {len(val_dataset)} (seed {args.seed})")

    pin_memory   = device.type == "cuda"
    loader_gen   = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory, generator=loader_gen)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    model     = get_model(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer)

    start_epoch   = 0
    best_acc      = 0.0
    best_epoch    = 0
    mlflow_run_id = None
    resumed       = False

    # ── Resume ────────────────────────────────────────────────────
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint  = torch.load(args.resume, map_location=device)
            # 舊版 checkpoint 沒有 scheduler_name，當時一律使用 StepLR
            ckpt_sched  = checkpoint.get("scheduler_name", "step")
            if ckpt_sched != args.scheduler:
                raise ValueError(f"Checkpoint was trained with --scheduler {ckpt_sched}; "
                                 f"resume with the same --scheduler (got {args.scheduler})")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch   = checkpoint["epoch"] + 1
            best_acc      = checkpoint["best_acc"]
            best_epoch    = checkpoint.get("best_epoch", 0)
            mlflow_run_id = checkpoint.get("mlflow_run_id")
            resumed       = True
            if checkpoint.get("val_size") != args.val_size:
                print("  ⚠️  Checkpoint used a different validation set; its best_acc is not comparable")
            print(f"Resumed from epoch {start_epoch}, best accuracy {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # ── Training log：只在 Resume 時保留歷史，並截掉 checkpoint 之後的紀錄 ──
    log_path     = os.path.join(args.save_dir, "training_log.json")
    training_log = []
    if resumed and os.path.exists(log_path):
        with open(log_path) as f:
            training_log = [e for e in json.load(f) if e["epoch"] <= start_epoch]

    # ── MLflow Run（Resume 時接續原本的 run）────────────────────────
    run, continued = start_mlflow_run(mlflow_run_id)
    with run:
        mlflow_run_id = run.info.run_id

        if continued:
            mlflow.set_tag("resumed_at_epoch", start_epoch)
        else:
            # 記錄所有超參數（param 不可覆寫，所以只在新 run 記錄）
            params = {
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "momentum":     args.momentum,
                "weight_decay": args.weight_decay,
                "optimizer":    "SGD",
                "scheduler":    "CosineAnnealingLR" if args.scheduler == "cosine" else "StepLR",
                "seed":         args.seed,
                "val_size":     args.val_size,
                "model":        "ResNet-18",
                "dataset":      "CIFAR-100",
                "device":       str(device),
                "resume":       args.resume if args.resume else "None",
            }
            if args.scheduler == "step":
                params.update({"step_size": args.step_size, "gamma": args.gamma})
            mlflow.log_params(params)

        # ── Training loop ─────────────────────────────────────────
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch [{epoch + 1}/{args.epochs}]")
            # 每個 epoch 依 epoch 編號重設種子：中斷後 Resume 與一次跑完的結果相同
            set_seed(args.seed + epoch, args.deterministic)
            loader_gen.manual_seed(args.seed + epoch)
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
            val_loss,   val_acc   = validate(val_loader, model, criterion, device)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

            # 記錄每個 epoch 的指標
            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc, 4),
                "val_loss":   round(val_loss, 4),
                "val_acc":    round(val_acc, 4),
                "lr":         optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)

            is_best = val_acc > best_acc
            if is_best:
                best_acc   = val_acc
                best_epoch = epoch + 1

            # 先 step scheduler 再存檔，checkpoint 內的 scheduler 狀態才會對應下一個 epoch
            scheduler.step()

            # ── Checkpoint 儲存（永遠存在 save_dir 根目錄）────────────
            save_checkpoint({
                "epoch":          epoch,
                "state_dict":     model.state_dict(),
                "best_acc":       best_acc,
                "best_epoch":     best_epoch,
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "scheduler_name": args.scheduler,
                "val_size":       args.val_size,
                "mlflow_run_id":  mlflow_run_id,
            }, is_best, args.save_dir)

            # 確認 checkpoint 有存到（防止 Drive 同步延遲）
            ckpt_path = os.path.join(args.save_dir, "checkpoint.pth")
            if not os.path.exists(ckpt_path):
                print(f"  ⚠️  WARNING: checkpoint.pth not found after save!")

            # ── 寫入 training log ──────────────────────────────────
            training_log.append({
                "epoch":      epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc, 4),
                "val_loss":   round(val_loss, 4),
                "val_acc":    round(val_acc, 4),
            })
            with open(log_path, "w") as f:
                json.dump(training_log, f, indent=2)

        # 訓練完成，記錄最終結果
        final_metrics = {"best_val_acc": round(best_acc, 4)}
        if best_epoch:  # 舊版 checkpoint 沒記錄 best_epoch，且之後沒有進步時為 0（未知）
            final_metrics["best_epoch"] = best_epoch
        mlflow.log_metrics(final_metrics)

        # ── MLflow artifact：訓練完成後才 log，不影響根目錄的 checkpoint ──
        best_ckpt = os.path.join(args.save_dir, "model_best.pth")
        if os.path.exists(best_ckpt):
            mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
            print(f"✅ model_best.pth logged to MLflow artifacts")

        print(f"Training complete. Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch or 'unknown'})")
        print(f"MLflow Run ID: {mlflow_run_id}")


def train(loader, model, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, targets)
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.2f}%'})

    return loss_meter.avg, acc_meter.avg


def validate(loader, model, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            acc = accuracy(outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.2f}%'})

    return loss_meter.avg, acc_meter.avg


if __name__ == "__main__":
    main()