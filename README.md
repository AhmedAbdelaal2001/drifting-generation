# Drifting Playground — Generative Modeling via Drifting

A minimal, readable implementation of the *core training idea* from **Generative Modeling via Drifting**:
a one-step generator trained by regressing to a **stop-gradient drifted target** computed from a **drifting field**.

This repo focuses on **toy 2D datasets** (two moons, spirals, Gaussian mixture), so you can understand and experiment with drifting
before scaling up to more complex settings.

---

## What this implements

At each training step:

1. Sample noise `z ~ N(0, I)` and generate `x = fθ(z)`.
2. Sample positives `y_pos ~ pdata` and use negatives `y_neg ~ qθ` (usually `y_neg = x`).
3. Compute drifting field `V(x, y_pos, y_neg, T)` (Algorithm 2).
4. Build a frozen drifted target: `x_target = stopgrad(x + drift_scale * V)`.
5. Minimize `MSE(x, x_target)`.

This matches the paper’s toy setup (Algorithm 1 / Algorithm 2) in spirit, with the same “double softmax” normalization used in their implementation.

---

## Repo layout

- `notebooks/drifting_toy.ipynb`: interactive playground (generator + drift + training loop)
- `drifting_toy/data.py`: datasets and samplers (kept out of the notebook)
- `drifting_toy/plotting.py`: all plotting utilities
- `drifting_toy/io.py`: config loading (YAML/JSON)
- `configs/*.yaml`: experiment configs
- `scripts/train.py`: run training without the notebook (useful for reproducibility)

---

## Installation

```bash
git clone <your-repo-url>
cd drifting-playground
pip install -r requirements.txt

## Quickstart

### Option A: Notebook (recommended)
Open `notebooks/drifting_toy.ipynb`, set:

```python
CFG_PATH = "../configs/toy_moons.yaml"

### Option B: Script
python scripts/train.py --config configs/toy_moons.yaml


## Outputs

After training, you should see images saved under `logging.run_dir`, e.g.:

- `dataset_preview.png`
- `state_step_000001.png`, `state_step_000400.png`, ...
- `final.png`

## Configuration guide

The most important knobs are:

- `drift.T` (temperature): higher = smoother drift, often more stable
- `drift.drift_scale`: higher = faster movement, but can overshoot
- `training.n_pos`: more positives reduces drift estimation noise
- `training.batch_size`: more negatives when using `y_neg = x`

### Troubleshooting
- If training diverges / samples explode: increase `drift.T`, reduce `drift_scale`
- If training is slow: increase `drift_scale` slightly, increase `steps`
- If you see mode dropping in GMM: increase `n_pos` and/or batch size

## Results

### Two Moons
![Two moons result](assets/moons_final.png)

### Spirals
![Spirals result](assets/spirals_final.png)

## Relation to the paper

- Implements a toy 2D version of the training loop (Alg. 1-style) and drifting field (Alg. 2).
- Uses the “double softmax + geometric mean” affinity normalization.
- Not included: large-scale/image experiments, encoders/latents, etc.

## Reproducibility

Runs are seeded via `system.seed`. Note that CUDA operations can still be nondeterministic on some GPUs/drivers.

## Extending

- Add a dataset: implement a sampler in `drifting_toy/data.py`, then point `dataset.name` in a config.
- Change generator: edit `ToyGenerator` in the notebook or script.

## Citation / credit

If you use this repo, please cite the original paper: *Generative Modeling via Drifting*.
(This repo is an educational toy implementation and is not affiliated with the authors.)