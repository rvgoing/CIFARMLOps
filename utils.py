import os

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    _, preds = output.max(1)
    correct = preds.eq(target).sum().item()
    return 100.0 * correct / target.size(0)


def save_checkpoint(state: dict, is_best: bool, save_dir: str):
    filepath = os.path.join(save_dir, "checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, "model_best.pth")
        torch.save(state, best_path)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0
