import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from evaluate import compute_per_class_accuracy, compute_top1_top5, save_report
from model import get_model
from utils import AverageMeter, accuracy, save_checkpoint


def test_model_shape():
    model = get_model().eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 100)


def test_accuracy_and_weighted_meter():
    assert accuracy(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1])) == 100
    meter = AverageMeter()
    meter.update(2, 3)
    meter.update(6, 1)
    assert meter.avg == 3
    meter.reset()
    assert meter.count == meter.avg == 0


def test_checkpoint_best_is_preserved(tmp_path):
    save_checkpoint({'epoch': 0}, True, tmp_path)
    save_checkpoint({'epoch': 1}, False, tmp_path)
    assert torch.load(tmp_path / 'checkpoint.pth', weights_only=True)['epoch'] == 1
    assert torch.load(tmp_path / 'model_best.pth', weights_only=True)['epoch'] == 0


def test_sparse_evaluation(tmp_path):
    labels = np.array([0, 2])
    probs = np.zeros((2, 6))
    probs[0, 0] = probs[1, 2] = 1
    assert compute_top1_top5(labels, probs) == (100, 100)
    names = list('abcdef')
    result = compute_per_class_accuracy(labels, labels, names)
    assert result == dict(zip(names, [100, 0, 100, 0, 0, 0]))
    assert 'f' in save_report(labels, labels, names, tmp_path)


def test_train_validate_cpu():
    torch.manual_seed(7)
    model = nn.Linear(4, 6)
    loader = DataLoader(TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 2, 3])), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = model.weight.detach().clone()
    loss, acc = train.train(loader, model, nn.CrossEntropyLoss(), optimizer, torch.device('cpu'))
    assert np.isfinite(loss) and 0 <= acc <= 100
    assert not torch.equal(before, model.weight)
    before = model.weight.detach().clone()
    train.validate(loader, model, nn.CrossEntropyLoss(), torch.device('cpu'))
    assert torch.equal(before, model.weight)


@pytest.mark.parametrize('argv', [['--epochs', '0'], ['--batch-size', '0'], ['--num-workers', '-1'], ['--resume', 'missing.pth']])
def test_invalid_cli(argv):
    with pytest.raises(SystemExit):
        train.parse_args(argv)


def test_resume_scheduler_log_and_best_epoch(tmp_path, monkeypatch):
    # Exercise the real orchestration and local MLflow without dataset downloads.
    args = train.parse_args(['--epochs', '2', '--save-dir', str(tmp_path), '--mlflow-dir', str(tmp_path / 'mlruns'), '--num-workers', '0'])
    monkeypatch.setattr(train, 'parse_args', lambda: args)
    monkeypatch.setattr(train.datasets, 'CIFAR100', lambda **kw: TensorDataset(torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)))
    monkeypatch.setattr(train, 'get_model', lambda **kw: nn.Linear(4, 100))
    def fake_train(loader, model, criterion, optimizer, device):
        optimizer.zero_grad()
        model(torch.zeros(2, 4)).sum().backward()
        optimizer.step()
        return 1., 50.
    monkeypatch.setattr(train, 'train', fake_train)
    scores = iter([70., 60., 65.])
    monkeypatch.setattr(train, 'validate', lambda *a: (1., next(scores)))
    (tmp_path / 'training_log.json').write_text('[{"epoch": 99}]')
    train.main()
    ckpt = torch.load(tmp_path / 'checkpoint.pth', weights_only=True)
    assert ckpt['scheduler']['last_epoch'] == 2
    assert ckpt['best_epoch'] == 1
    assert len(json.loads((tmp_path / 'training_log.json').read_text())) == 2
    args.resume = str(tmp_path / 'checkpoint.pth')
    args.epochs = 3
    train.main()
    ckpt = torch.load(args.resume, weights_only=True)
    assert ckpt['scheduler']['last_epoch'] == 3
    assert ckpt['best_epoch'] == 1
    assert [r['epoch'] for r in json.loads((tmp_path / 'training_log.json').read_text())] == [1, 2, 3]
    from mlflow import MlflowClient
    runs = MlflowClient().search_runs([train.mlflow.get_experiment_by_name(args.exp_name).experiment_id])
    assert len(runs) == 2
    assert all(r.info.status == 'FINISHED' for r in runs)
    assert all(r.data.metrics['best_epoch'] == 1 for r in runs)


def test_evaluation_artifacts_and_checkpoint(tmp_path):
    from evaluate import load_model, run_inference, plot_confusion_matrix, plot_per_class_accuracy, plot_loss_curve, save_summary
    model = get_model().eval()
    save_checkpoint({'state_dict': model.state_dict(), 'epoch': 0, 'best_acc': 12.}, True, tmp_path)
    restored = load_model(tmp_path / 'model_best.pth', 100, torch.device('cpu'))
    loader = DataLoader(TensorDataset(torch.zeros(2, 3, 32, 32), torch.tensor([0, 2])), batch_size=2)
    labels, preds, probs = run_inference(loader, restored, torch.device('cpu'))
    assert probs.shape == (2, 100)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    names = [str(i) for i in range(100)]
    per_class = compute_per_class_accuracy(labels, preds, names)
    plot_confusion_matrix(labels, preds, names, tmp_path)
    plot_per_class_accuracy(per_class, tmp_path)
    log = tmp_path / 'training_log.json'
    log.write_text(json.dumps([dict(epoch=1, train_loss=1., val_loss=2., train_acc=10., val_acc=12.)]))
    plot_loss_curve(log, tmp_path)
    save_summary(*compute_top1_top5(labels, probs), per_class, tmp_path)
    for name in ['confusion_matrix.png', 'per_class_accuracy.png', 'loss_acc_curve.png', 'summary.json']:
        assert (tmp_path / name).stat().st_size > 0


def test_demo_cpu_predict(monkeypatch):
    for name in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "all_proxy", "https_proxy", "http_proxy"):
        monkeypatch.delenv(name, raising=False)
    import demo
    from PIL import Image
    monkeypatch.setattr(demo, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(demo, 'MODEL', get_model().eval())
    assert demo.predict(None) == {}
    result = demo.predict(Image.new('RGB', (64, 64)))
    assert len(result) == 5
    assert all(0 <= prob <= 1 for prob in result.values())
