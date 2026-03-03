from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from drifting_toy.io import load_config
from drifting_toy.utils import set_global_seed, get_device
from drifting_toy.data import make_real_sampler
from drifting_toy.plotting import plot_training_state, plot_dataset_preview, save_fig


class ToyGenerator(nn.Module):
    def __init__(self, *, z_dim: int, hidden_dim: int, n_layers: int, out_dim: int):
        super().__init__()
        self.z_dim = int(z_dim)
        layers: list[nn.Module] = []
        d = self.z_dim
        for _ in range(int(n_layers)):
            layers += [nn.Linear(d, int(hidden_dim)), nn.SiLU()]
            d = int(hidden_dim)
        layers += [nn.Linear(d, int(out_dim))]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def sample_z(self, n: int, *, device: torch.device) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@torch.no_grad()
def compute_V(x, y_pos, y_neg, *, T: float, ignore_self_in_y_neg: bool, eps: float = 1e-12):
    N = x.shape[0]
    dist_pos = torch.cdist(x, y_pos)
    dist_neg = torch.cdist(x, y_neg)
    if ignore_self_in_y_neg and y_neg.shape[0] == N:
        dist_neg = dist_neg + torch.eye(N, device=x.device, dtype=x.dtype) * 1e6

    logit = torch.cat([-dist_pos / T, -dist_neg / T], dim=1)
    A_row = torch.softmax(logit, dim=-1)
    A_col = torch.softmax(logit, dim=-2)
    A = torch.sqrt(torch.clamp(A_row * A_col, min=eps))

    N_pos = y_pos.shape[0]
    A_pos = A[:, :N_pos]
    A_neg = A[:, N_pos:]

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    return (W_pos @ y_pos) - (W_neg @ y_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg["logging"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg["system"])
    set_global_seed(cfg["system"]["seed"])

    sample_real = make_real_sampler(cfg, device=device)

    fig = plot_dataset_preview(sample_real, n=cfg["logging"]["n_preview"])
    save_fig(fig, run_dir / "dataset_preview.png")

    gen_cfg = cfg["generator"]
    train_cfg = cfg["training"]
    drift_cfg = cfg["drift"]

    gen = ToyGenerator(
        z_dim=gen_cfg["z_dim"],
        hidden_dim=gen_cfg["hidden_dim"],
        n_layers=gen_cfg["n_layers"],
        out_dim=gen_cfg["out_dim"],
    ).to(device)

    opt = torch.optim.AdamW(gen.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    losses, mean_v_norms = [], []
    for step in range(1, int(train_cfg["steps"]) + 1):
        z = gen.sample_z(int(train_cfg["batch_size"]), device=device)
        x = gen(z)
        y_pos = sample_real(int(train_cfg["n_pos"]))
        y_neg = x

        with torch.no_grad():
            V = compute_V(x, y_pos, y_neg, T=float(drift_cfg["T"]), ignore_self_in_y_neg=True)
            x_target = (x + float(drift_cfg["drift_scale"]) * V).detach()

        loss = F.mse_loss(x, x_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        mean_v_norms.append(float(V.norm(dim=1).mean().item()))

        if step % int(cfg["logging"]["print_every"]) == 0:
            print(f"step {step:6d} | loss={losses[-1]:.6f} | mean||V||={mean_v_norms[-1]:.6f}")

        if step == 1 or step % int(cfg["logging"]["plot_every"]) == 0:
            gen.eval()
            fig = plot_training_state(
                sample_real=sample_real,
                sample_fake=lambda n: gen(gen.sample_z(n, device=device)).detach(),
                losses=losses,
                mean_v_norms=mean_v_norms,
                step=step,
                n_vis=int(cfg["logging"]["n_vis"]),
            )
            save_fig(fig, run_dir / f"state_step_{step:06d}.png")
            gen.train()

    gen.eval()
    fig = plot_training_state(
        sample_real=sample_real,
        sample_fake=lambda n: gen(gen.sample_z(n, device=device)).detach(),
        losses=losses,
        mean_v_norms=mean_v_norms,
        step=int(train_cfg["steps"]),
        n_vis=int(cfg["logging"]["n_vis"]),
    )
    save_fig(fig, run_dir / "final.png")
    print("Done. Outputs in:", run_dir.resolve())


if __name__ == "__main__":
    main()