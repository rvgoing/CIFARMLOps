"""
evaluate.py
-----------
訓練完成後的完整評量模組。

使用方式：
    python evaluate.py --checkpoint ./checkpoints/model_best.pth
    python evaluate.py --checkpoint ./checkpoints/model_best.pth --save-dir ./eval_results
"""

import argparse
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use('Agg')  # 無 GUI 環境（Colab / Server）使用
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from model import get_model


# CIFAR-100 的 100 個類別名稱
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm',
]


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 Evaluation")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", default="./data", type=str, help="Path to CIFAR-100 dataset")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--save-dir", default="./eval_results", type=str, help="Directory to save evaluation results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--num-classes", default=100, type=int, help="Number of classes in the model")
    return parser.parse_args()


def load_model(checkpoint_path, num_classes, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    model = get_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    best_acc = checkpoint.get("best_acc", "N/A")
    epoch    = checkpoint.get("epoch", "N/A")
    if isinstance(best_acc, float):
        print(f"  Loaded from epoch {epoch}, best_acc={best_acc:.2f}%")
    else:
        print(f"  Loaded epoch {epoch}")
    return model


def get_dataloader(data_dir, batch_size, num_workers):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def run_inference(loader, model, device):
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1)

            all_labels.append(targets.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def compute_top1_top5(labels, probs):
    top1 = top_k_accuracy_score(labels, probs, k=1) * 100
    top5 = top_k_accuracy_score(labels, probs, k=5) * 100
    return top1, top5


def compute_per_class_accuracy(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    return {class_names[i]: round(per_class[i], 2) for i in range(len(class_names))}


def plot_loss_curve(log_path, save_dir):
    if not os.path.exists(log_path):
        print(f"  [Skip] No training log found at {log_path}")
        return

    with open(log_path) as f:
        logs = json.load(f)

    epochs     = [d["epoch"]      for d in logs]
    train_loss = [d["train_loss"] for d in logs]
    val_loss   = [d["val_loss"]   for d in logs]
    train_acc  = [d["train_acc"]  for d in logs]
    val_acc    = [d["val_acc"]    for d in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", marker='o', markersize=3)
    ax1.plot(epochs, val_loss,   label="Val Loss",   marker='o', markersize=3)
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="Train Acc", marker='o', markersize=3)
    ax2.plot(epochs, val_acc,   label="Val Acc",   marker='o', markersize=3)
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "loss_acc_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(labels, preds, class_names, save_dir, top_n=20):
    cm = confusion_matrix(labels, preds)
    errors  = cm.sum(axis=1) - cm.diagonal()
    top_idx = np.argsort(errors)[-top_n:][::-1]
    cm_sub  = cm[np.ix_(top_idx, top_idx)]
    sub_names = [class_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_sub, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(top_n))
    ax.set_yticks(range(top_n))
    ax.set_xticklabels(sub_names, rotation=90, fontsize=8)
    ax.set_yticklabels(sub_names, fontsize=8)
    ax.set_title(f"Confusion Matrix (Top-{top_n} Most Confused Classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_class_accuracy(per_class_acc, save_dir, bottom_n=20):
    sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1])
    worst  = sorted_items[:bottom_n]
    names  = [x[0] for x in worst]
    values = [x[1] for x in worst]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names, values, color='steelblue')
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Bottom-{bottom_n} Per-Class Accuracy")
    ax.set_xlim(0, 100)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "per_class_accuracy.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def save_report(labels, preds, class_names, save_dir):
    report   = classification_report(labels, preds, target_names=class_names, digits=4)
    out_path = os.path.join(save_dir, "classification_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"  Saved: {out_path}")
    return report


def save_summary(top1, top5, per_class_acc, save_dir):
    worst5 = sorted(per_class_acc.items(), key=lambda x: x[1])[:5]
    best5  = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)[:5]
    summary = {
        "top1_accuracy":   round(top1, 4),
        "top5_accuracy":   round(top5, 4),
        "best_5_classes":  {k: v for k, v in best5},
        "worst_5_classes": {k: v for k, v in worst5},
    }
    out_path = os.path.join(save_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")
    return summary


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    class_names = CIFAR100_CLASSES[:args.num_classes]

    print("\n========== CIFAR-100 Evaluation ==========")

    print("[1/5] Loading model ...")
    model = load_model(args.checkpoint, args.num_classes, device)

    print("[2/5] Running inference ...")
    loader = get_dataloader(args.data_dir, args.batch_size, args.num_workers)
    labels, preds, probs = run_inference(loader, model, device)

    print("[3/5] Computing metrics ...")
    top1, top5 = compute_top1_top5(labels, probs)
    print(f"      Top-1 Accuracy : {top1:.2f}%")
    print(f"      Top-5 Accuracy : {top5:.2f}%")
    per_class_acc = compute_per_class_accuracy(labels, preds, class_names)

    print("[4/5] Saving plots ...")
    log_path = os.path.join(os.path.dirname(args.checkpoint), "training_log.json")
    plot_loss_curve(log_path, args.save_dir)
    plot_confusion_matrix(labels, preds, class_names, args.save_dir)
    plot_per_class_accuracy(per_class_acc, args.save_dir)

    print("[5/5] Saving reports ...")
    save_report(labels, preds, class_names, args.save_dir)
    summary = save_summary(top1, top5, per_class_acc, args.save_dir)

    # 儲存推論結果，供後續分析使用（不需要重跑模型）
    np.save(os.path.join(args.save_dir, 'labels.npy'), labels)
    np.save(os.path.join(args.save_dir, 'preds.npy'),  preds)
    print(f"  Saved: labels.npy / preds.npy")

    print("\n========== Summary ==========")
    print(f"Top-1 Accuracy : {top1:.2f}%")
    print(f"Top-5 Accuracy : {top5:.2f}%")
    print(f"\nBest 5  : { {k: f'{v}%' for k, v in list(summary['best_5_classes'].items())} }")
    print(f"Worst 5 : { {k: f'{v}%' for k, v in list(summary['worst_5_classes'].items())} }")
    print(f"\nAll results saved to: {args.save_dir}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
