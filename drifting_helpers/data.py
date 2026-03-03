from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch

# ----------------------------
# Samplers (NumPy)
# ----------------------------

def sample_two_moons(n: int, *, noise: float, distance: float, rng: np.random.Generator) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1

    t1 = rng.uniform(0.0, np.pi, size=n1)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)

    t2 = rng.uniform(0.0, np.pi, size=n2)
    x2 = np.stack([1.0 - np.cos(t2), -np.sin(t2) - distance], axis=1)

    x = np.concatenate([x1, x2], axis=0)
    if noise > 0:
        x = x + rng.normal(0.0, noise, size=x.shape)
    return x.astype(np.float32)

def sample_two_spirals(
    n: int, *, noise: float, turns: float, radius: float, gap: float, rng: np.random.Generator
) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    theta_max = turns * 2.0 * np.pi

    t1 = rng.uniform(0.0, theta_max, size=n1)
    t2 = rng.uniform(0.0, theta_max, size=n2)

    r1 = (t1 / theta_max) * radius
    r2 = (t2 / theta_max) * radius

    x1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)
    x2 = np.stack([r2 * np.cos(t2 + np.pi), r2 * np.sin(t2 + np.pi)], axis=1)

    if gap != 0.0:
        x2 = x2 * (1.0 + gap)

    x = np.concatenate([x1, x2], axis=0)
    if noise > 0:
        x = x + rng.normal(0.0, noise, size=x.shape)
    return x.astype(np.float32)

def _random_spd_2x2(rng: np.random.Generator, *, cov_scale: float, anisotropy: float, random_rotation: bool) -> np.ndarray:
    base = cov_scale * rng.uniform(0.6, 1.4)
    aspect = rng.uniform(1.0, anisotropy)
    eigvals = np.array([base, base * aspect], dtype=np.float64)

    if random_rotation:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
    else:
        R = np.eye(2, dtype=np.float64)

    cov = R @ np.diag(eigvals) @ R.T
    return cov

def build_gaussians_on_circle(cfg: dict, rng: np.random.Generator) -> dict:
    k = int(cfg["gaussians_k"])
    radius = float(cfg["gaussians_circle_radius"])
    cov_scale = float(cfg["gaussians_cov_scale"])
    anisotropy = float(cfg["gaussians_cov_anisotropy"])
    random_rotation = bool(cfg["gaussians_cov_rot"])

    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float64)

    covs = np.stack(
        [
            _random_spd_2x2(rng, cov_scale=cov_scale, anisotropy=anisotropy, random_rotation=random_rotation)
            for _ in range(k)
        ],
        axis=0,
    ).astype(np.float64)

    w_mode = str(cfg.get("gaussians_weights", "uniform")).lower()
    if w_mode == "uniform":
        weights = np.ones(k, dtype=np.float64) / k
    elif w_mode == "random":
        weights = rng.random(k).astype(np.float64)
        weights = weights / weights.sum()
    else:
        raise ValueError('gaussians_weights must be "uniform" or "random"')

    # optional far component
    if bool(cfg.get("gaussians_add_far_component", False)):
        far_r = float(cfg.get("gaussians_far_radius", 12.0))
        far_angle = cfg.get("gaussians_far_angle", 0.0)
        far_angle = float(rng.uniform(0.0, 2.0 * np.pi)) if far_angle is None else float(far_angle)

        far_mean = np.array([far_r * np.cos(far_angle), far_r * np.sin(far_angle)], dtype=np.float64)
        far_cov = _random_spd_2x2(
            rng,
            cov_scale=float(cfg.get("gaussians_far_cov_scale", cov_scale)),
            anisotropy=anisotropy,
            random_rotation=bool(cfg.get("gaussians_cov_rot", True)),
        ).astype(np.float64)

        far_weight = float(cfg.get("gaussians_far_weight", 0.08))
        if not (0.0 < far_weight < 1.0):
            raise ValueError("gaussians_far_weight must be in (0, 1)")

        means = np.concatenate([means, far_mean[None, :]], axis=0)
        covs = np.concatenate([covs, far_cov[None, :, :]], axis=0)

        weights = weights * (1.0 - far_weight)
        weights = np.concatenate([weights, np.array([far_weight], dtype=np.float64)], axis=0)

    return {"means": means, "covs": covs, "weights": weights}

def sample_gaussians(n: int, mixture: dict, rng: np.random.Generator) -> np.ndarray:
    means = mixture["means"]
    covs = mixture["covs"]
    weights = mixture["weights"]

    k = means.shape[0]
    comp = rng.choice(k, size=n, p=weights)

    x = np.empty((n, 2), dtype=np.float64)
    for j in range(k):
        idx = np.where(comp == j)[0]
        if idx.size == 0:
            continue
        x[idx] = rng.multivariate_normal(mean=means[j], cov=covs[j], size=idx.size)
    return x.astype(np.float32)

# ----------------------------
# Public interface used by notebook/scripts
# ----------------------------

def make_real_sampler(cfg: dict, *, device: torch.device) -> callable:
    """
    Returns a function sample_real(n) -> torch.Tensor [n,2] on `device`.
    """
    dataset = str(cfg["dataset"]["name"]).lower()
    seed = int(cfg["system"]["seed"])
    rng = np.random.default_rng(seed)

    gmm_cache = None
    if dataset == "gaussians":
        gmm_cache = build_gaussians_on_circle(cfg["dataset"], rng=rng)

    def sample_real(n: int) -> torch.Tensor:
        nonlocal gmm_cache
        if dataset == "moons":
            x = sample_two_moons(
                n,
                noise=float(cfg["dataset"]["moons_noise"]),
                distance=float(cfg["dataset"]["moons_distance"]),
                rng=rng,
            )
        elif dataset == "spirals":
            x = sample_two_spirals(
                n,
                noise=float(cfg["dataset"]["spirals_noise"]),
                turns=float(cfg["dataset"]["spirals_turns"]),
                radius=float(cfg["dataset"]["spirals_radius"]),
                gap=float(cfg["dataset"]["spirals_gap"]),
                rng=rng,
            )
        elif dataset == "gaussians":
            assert gmm_cache is not None
            x = sample_gaussians(n, mixture=gmm_cache, rng=rng)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        return torch.from_numpy(x).to(device)

    return sample_real