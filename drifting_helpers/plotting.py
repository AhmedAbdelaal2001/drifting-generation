from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_fig(fig, path: str | Path, *, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

@torch.no_grad()
def plot_dataset_preview(sample_real, *, n: int = 3000):
    x = sample_real(n).detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.scatter(x[:, 0], x[:, 1], s=6, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("Dataset preview")
    return fig

@torch.no_grad()
def plot_training_state(
    *,
    sample_real,
    sample_fake,
    losses: list[float],
    mean_v_norms: list[float],
    step: int,
    n_vis: int = 2000,
):
    real = sample_real(n_vis).detach().cpu().numpy()
    fake = sample_fake(n_vis).detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(real[:, 0], real[:, 1], s=6, alpha=0.35, label="real")
    ax1.scatter(fake[:, 0], fake[:, 1], s=6, alpha=0.35, label="gen")
    ax1.set_title(f"Step {step}: real vs gen")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right")

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(losses, linewidth=1.5)
    ax2.set_title("Training loss: MSE(x, stopgrad(x + drift_scale·V))")
    ax2.set_xlabel("iteration")
    ax2.grid(True, alpha=0.2)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(mean_v_norms, linewidth=1.5)
    ax3.set_title("Mean ||V|| over batch (pre-scale)")
    ax3.set_xlabel("iteration")
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig