# Repository review (2026-10-10)

Reviewed every tracked file, including all Colab notebook cells, at main
40919655d725c46fdd833168d6c241f90eada6a2. The original repository has five
Python modules, one notebook, requirements, README, changelog and gitignore;
there are no tests, CI workflows, packaging metadata or declared Python range.

## Findings addressed

- Checkpoints saved before scheduler.step() diverge from uninterrupted training.
  Version 2 saves post-step state; legacy states receive the missing step.
- best_epoch previously recorded the last epoch. It now follows improvements.
- Fresh runs inherited unrelated log history. Only resumes reuse/truncate logs.
- Missing resume files silently started fresh training. CLI now fails early.
- Evaluation inferred class IDs from observed labels, breaking sparse subsets.
  Metrics and plots use explicit IDs; absent classes use zero accuracy/support.
- Checkpoint loads use weights_only=True; directory tracking uses portable URIs.
- Current MLflow requires explicit file-store opt-in, handled in local mode.
- Unbounded dependencies and per-package notebook installs allowed inconsistent
  resolution. Major-version bounds, NumPy <2 and batch notebook installation
  with existing torch/torchvision constraints reduce this risk.
- Secrets, tracking files, evaluation outputs and model exports are ignored.
- CPU regression tests and Python 3.10/3.11 Actions were added.

## Remaining limitations

CIFAR's test split is still used for epoch selection; a separate validation split
is needed before reporting an unbiased final test score. Seed/RNG restoration,
atomic checkpoint writes, gradient accumulation and VRAM monitoring remain
future work. Dependency ranges are bounded, not a fully reproducible lockfile.
No production deployment was implemented. CUDA/GTX 1050 and full dataset
training were not run here. Colab cells were inspected, not executed in Colab.
Interactive Gradio UI and remote MLflow authentication are not end-to-end tested.
Use a fresh save directory per experiment to avoid stale best-model artifacts.

## Validation

Local isolated `.venv`: Python 3.12.14, torch 2.2.2+cpu,
torchvision 0.17.2+cpu, MLflow 3.17.0. 12 tests passed in 11.48s.
Statement coverage across all five modules: 74% (train 94%, evaluate 72%,
model 80%, utils 100%, demo 34%). `pip check` and `git diff --check` pass.
Python 3.10/3.11 are configured in CI; their results are separate from this
local Python 3.12 verification. Telemetry is disabled before dependency imports
in tests and CI. No CIFAR data, secrets or model binaries are committed.
