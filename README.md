# CS 4644/7643 · Neural PDE Option Greeks

Physics-Informed Neural Networks (PINNs) allow us to embed financial theory directly into a neural network’s training objective. This repository contains the full research stack for our Georgia Tech Deep Learning project on **efficiently estimating option Greeks by solving the Black–Scholes PDE with a neural surrogate**. It includes data generation, model code, adaptive sampling utilities, notebooks for diagnostics, and project documentation.

---

## 1. Motivation and Research Goals

- **Theory-aware learning:** Rather than training a model purely on labelled prices, we minimise the Black–Scholes residual during optimisation. The network therefore respects absence-of-arbitrage dynamics.
- **All Greeks from one model:** By treating volatility σ as an input alongside stock price S and time t, the PINN learns a smooth surface $V_\theta(S,t,\sigma)$. Once trained, $\Delta, \Gamma, \theta, \nu$, and $\rho$ are obtained via automatic differentiation without retraining.
- **Efficiency vs. classical methods:** Finite differences are biased/noisy for higher-order Greeks, and Monte Carlo requires millions of paths. A trained PINN evaluates in milliseconds while remaining faithful to analytic solutions.
- **Project objectives**
  1. Demonstrate a single PINN that generalises across volatility regimes.
  2. Benchmark it against analytic, finite-difference, and Monte Carlo baselines.
  3. Investigate adaptive sampling strategies that concentrate learning on regions with high PDE residual.

---

## 2. Repository Overview

```
Neural-PDE-Option-Greeks/
├── README.md
├── requirements.txt
├── proposal.md                 # Initial pitch
├── project_board.md            # Task tracker
│
├── src/                        # Importable package (`import src`)
│   ├── __init__.py             # Shared path helpers (DATA_DIR, RESULTS_DIR, …)
│   ├── preprocessing.py        # Log-moneyness / time-to-maturity normalisation
│   ├── utils/
│   │   └── black_scholes.py    # Closed-form Black–Scholes price & Greeks
│   ├── baselines/
│   │   ├── finite_difference.py
│   │   └── monte_carlo.py
│   ├── models/
│   │   └── pinn_model.py       # Residual MLP backbone
│   ├── losses.py               # Physics-informed loss & PDE residual
│   ├── data.py                 # Synthetic dataset generation (train/val/test)
│   ├── train.py                # Training loop with warmup & adaptive sampling
│   ├── eval.py                 # Price-surface visualisation
│   └── test.py                 # Out-of-sample benchmarking CLI
│
├── notebooks/
│   ├── Sanity_Check.ipynb      # Lightweight smoke test
│   ├── System_Stress_Test.ipynb# Regression notebook with configurable experiments
│   └── End_to_End_Evaluation.ipynb # Full training/validation/testing pipeline
│
├── data/                       # Generated `.npy` datasets (ignored by git)
├── results/                    # Checkpoints & JSON summaries
├── figures/                    # Generated plots (price surfaces, residuals, etc.)
└── reports/                    # Milestone/final write-ups & poster drafts
```

---

## 3. Theoretical Background

The Black–Scholes PDE for a European option with price function $V(S,t,\sigma)$ is

$$
\partial_t V + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS}V + rS\partial_S V - rV = 0,
$$

with terminal payoff $V(T,S,\sigma) = \max(S - K, 0)$. In our PINN:

- **Input features:** $(S, t, \sigma)$ transformed into log-moneyness $x = \log(S/K)$, time-to-maturity $\tau = T - t$, and scaled to $[-1,1]$.
- **Network:** Residual MLP (default 5×128) mapping features → scalar price $V_\theta$.
- **Loss terms:**
  - $L_{\text{price}} = \mathbb{E}[(V_\theta - V_{BS})^2]$,
  - $L_{\text{PDE}} = \mathbb{E}[(\partial_t V_\theta + \ldots - rV_\theta)^2]$,
  - $L_{\text{reg}} = \lambda \mathbb{E}[(\partial_S V_\theta)^2]$ (Sobolev regulariser for smooth $\Delta$).
- **Automatic differentiation** provides $\partial_S V_\theta$, $\partial^2_{SS} V_\theta$, and $\partial_t V_\theta$ exactly, enabling reliable PDE residuals and Greek extraction.

---

## 4. Environment Setup

1. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **GPU users:** Install the CUDA-enabled PyTorch wheel matching your driver/toolkit, or let `pip install torch` select the CPU build.

3. **Directory permissions:** Matplotlib warnings about non-writable cache directories can be resolved by setting `export MPLCONFIGDIR=$PWD/.mpl` before running scripts.

---

## 5. Data Generation & Preprocessing

- Synthetic datasets are created from the analytic Black–Scholes formula with:
  - $S \sim \text{Uniform}(20, 200)$,
  - $t \sim \text{Uniform}(0.01, 2.0)$,
  - $\sigma \sim \text{Uniform}(0.05, 0.6)$.
- We save `[S, t, σ, V]` tuples as `.npy` arrays in `data/`.
- Preprocessing (implemented in `src.preprocessing.normalize_inputs`) converts raw inputs to log-moneyness and scaled time/volatility before they reach the network.

To (re)generate the datasets (train/val/test) from the command line:

```bash
python -m src.data \
  --n-train 1000000 \
  --n-val 100000 \
  --n-test 100000 \
  --seed 123
```

This writes `synthetic_train.npy`, `synthetic_val.npy`, and `synthetic_test.npy` into `data/`. Pass `--visualize` to emit interactive summary dashboards (`*.html`) under `<output-dir>/figures/` for quick sanity checks.

---

## 6. Training the PINN

`src.train.train` exposes the full training loop. You can call it from a Python script, the REPL, or a notebook.

```python
from src.train import train

model, history = train(
    epochs=50,
    lr=1e-3,
    batch_size=4096,
    device="cuda",             # "cpu" or "cuda"
    use_warmup=True,
    warmup_steps=500,
    grad_clip=1.0,
    adaptive_sampling=True,
    adaptive_every=5,
    adaptive_points=10_000,
    adaptive_radius=0.1,
    adaptive_eval_samples=50_000,
    checkpoint_path="results/pinn_checkpoint.pt",
    save_checkpoint=True,
)
```

### Command-line usage

```bash
python -m src.train \
    --epochs 25 \
    --lr 5e-4 \
    --batch-size 2048 \
    --device cuda \
    --adaptive-sampling \
    --adaptive-every 5 \
    --warmup-steps 300 \
    --checkpoint-path results/pinn_checkpoint.pt \
    --lambda-reg 0.01
```

Use `--no-warmup` to disable the linear schedule and `--no-save-checkpoint` to skip writing weights. Add `--no-val` to skip validation or `--no-plots` to avoid saving charts. Run `python -m src.train -h` for the full list of options.

By default training writes `results/training_history.json` and saves both linear and log-scale loss plots (`loss_curves.png`, `loss_curves_log.png`) under `figures/training_curves/`. Supply `--log-path`, `--plot-path`, or `--lambda-reg` to customise outputs.

### Key training features

- **Warmup scheduling:** When `use_warmup=True`, the learning rate ramps up linearly during the first `warmup_steps` optimiser steps.
- **Adaptive sampling:** Periodically identifies collocation points with the largest PDE residuals, jitter-resamples their neighbourhoods, and augments the dataset. This stabilises learning in high-curvature regions that drive Γ.
- **Validation loop:** When a validation set is provided, per-epoch metrics (`val_loss`, `val_pde`, …) are logged alongside training values.
- **Device selection:** Pass `"cuda"` to leverage a GPU; the helper automatically falls back to CPU if CUDA is unavailable.
- **Logging & plots:** History is stored as JSON and loss curves are rendered automatically (configurable via CLI flags).
- **Regularization control:** Tune `--lambda-reg` to scale the gradient-squared penalty term.
- **Checkpointing:** Controlled via `save_checkpoint` / `load_checkpoint`.

---

## 7. Evaluating Models & Baselines

1. **Monte Carlo / finite-difference baselines**

   ```python
   from src.baselines import finite_diff_greeks, mc_pathwise_greeks

   S = 100.0
   fd = finite_diff_greeks(S)
   mc = mc_pathwise_greeks(S, seed=0, N=50_000)
   print(fd["delta"], fd["gamma"], fd["theta"], fd["vega"], fd["rho"])
   print(mc["delta"], mc["theta"], mc["vega"], mc["rho"])
   ```

   The Monte Carlo helper accepts either `seed` or a NumPy generator for reproducibility.

2. **Price surface visualisation**

   ```bash
   python -m src.eval
   ```

   Customise the device, grid resolution, and export path:

   ```bash
   python -m src.eval \
       --model-path results/pinn_checkpoint.pt \
       --output-path figures/final_results/pinn_surface \
       --device cuda \
       --grid-points 150
   ```

   This loads `results/pinn_checkpoint.pt`, samples a grid over $S \in [20, 200]$ and $σ \in [0.05, 0.6]$, normalises the inputs, and emits both contour and 3D surface plots (look for `*_contour.png` and `*_3d.png` in `figures/final_results/`).

3. **Out-of-sample evaluation**

   ```bash
   python -m src.test \
       --data-path data/synthetic_test.npy \
       --model-path results/pinn_checkpoint.pt \
       --device cuda \
       --sample-size 5000 \
       --mc-paths 50000 \
       --output results/oos_metrics.json \
       --visualize \
       --surface-grid 40
   ```

   The testing script compares the PINN against Black–Scholes analytics, finite differences, and Monte Carlo on a held-out set, printing MAE/RMSE diagnostics while saving both a JSON report and summary plots (default `results/figures/oos/`) including 2D overlays and 3D surfaces for price, Δ, and Γ (analytic vs. PINN plus error). Adjust Monte Carlo paths, subsample size, or seeds to probe robustness.

4. **Validation metrics**
   The `System_Stress_Test.ipynb` notebook:
   - reports dataset statistics,
   - benchmarks baselines vs. analytic Greeks,
   - runs training with your chosen configuration,
   - computes MAE/RMSE for price and Greeks on a validation subset,
   - renders PDE residual heatmaps & price contours,
   - exports a JSON summary to `results/stress_test_summary.json`.

---

## 8. Notebooks

- **Sanity_Check.ipynb** — lightweight smoke test to ensure imports, dataset generation, baselines, and a mini PINN run all work in your environment.
- **System_Stress_Test.ipynb** — a regression harness with a CONFIG cell controlling device, warmup, adaptive sampling, batch size, checkpoint behaviour, and evaluation sample counts.
- **End_to_End_Evaluation.ipynb** — fully scripted pipeline that generates data, trains the PINN, evaluates on validation data, and benchmarks out-of-sample performance with visualizations.

Both notebooks keep outputs inside the repo’s `figures/` and `results/` directories so they can be versioned or cleaned easily.

---

## 9. Project Status & Next Steps

Track progress in `project_board.md`. Completed milestones include:

- ✅ Synthetic dataset generation & preprocessing
- ✅ PINN architecture with physics-informed loss
- ✅ Baselines (finite difference / Monte Carlo) with reproducible seeds
- ✅ Adaptive sampling prototype and logging infrastructure
- ✅ Stress-test notebook covering the entire pipeline

Upcoming focus areas:

- Full RMSE / runtime benchmarking
- Hyperparameter sweeps (Sobolev λ, depth, warmup schedules)
- Milestone report drafting and figure polishing
