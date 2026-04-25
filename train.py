import argparse
import json
import os
import warnings

from tqdm import tqdm

# Suppress NumPy 2.4+ deprecation warning from torchvision CIFAR loader
warnings.filterwarnings('ignore', message='.*dtype.*align should be passed as Python or NumPy boolean.*')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model
from utils import accuracy, save_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 training with PyTorch")
    parser.add_argument("--data-dir", default="./data", type=str, help="Path to CIFAR-100 dataset folder")
    parser.add_argument("--epochs", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", default=128, type=int, help="Batch size for training and validation")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--num-workers", default=4, type=int, help="Data loader worker count")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to train on")
    parser.add_argument("--save-dir", default="./checkpoints", type=str, help="Directory to save models and logs")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

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

    train_dataset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    model = get_model(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    start_epoch = 0
    best_acc    = 0.0

    # ── 讀取已有的 training log（Resume 時保留歷史紀錄）──────────────
    log_path     = os.path.join(args.save_dir, "training_log.json")
    training_log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)

    # ── Resume ────────────────────────────────────────────────────
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc    = checkpoint["best_acc"]
            print(f"Resumed from epoch {start_epoch}, best accuracy {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        val_loss,   val_acc   = validate(val_loader, model, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        is_best  = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint({
            "epoch":      epoch,
            "state_dict": model.state_dict(),
            "best_acc":   best_acc,
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
        }, is_best, args.save_dir)

        # ── 寫入 training log ─────────────────────────────────────
        training_log.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        })
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        scheduler.step()

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")


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