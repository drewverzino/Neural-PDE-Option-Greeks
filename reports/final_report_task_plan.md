# Final Report Task Plan

## Core experiments
- Main PINN run (seeded) with locked hyperparameters; record price RMSE, MAE for Delta/Gamma/Theta/Vega/Rho, Gamma total variation, and PDE residual stats; save training curves.
- Out-of-distribution vols: evaluate sigma in [0.60, 0.65] with the same metrics and report ratio vs in-distribution error.
- Stability: repeat best config across 3 seeds; report mean/std; keep checkpoints and metrics CSVs.
- Runtime: measure inference latency (CPU and GPU) for single sample and batch; log training throughput.

## Ablations
- Supervised-only MLP (no PDE term) vs PINN.
- With vs without adaptive sampling.
- Sobolev lambda sweep {0.001, 0.01, 0.1}; optional Gamma supervision variant.
- Fixed-sigma PINN vs sigma-input PINN.
- Depth variations (3/5/7 layers) if time; track accuracy vs latency.

## Baselines
- Finite-difference Greeks with consistent bumps; log MAE vs analytic on the held-out test set and runtime.
- Monte Carlo pathwise Greeks with fixed path count and seed; log MAE and runtime.
- Ensure parameter ranges match training data; store outputs in `results/` as CSV with config notes.

## Figures and tables
- Surfaces and error heatmaps for price/Delta/Gamma/Theta/Vega; Gamma smoothness plot (TV over grid); PDE residual heatmap; training loss curves; OOD error vs sigma plot.
- Tables: main metrics, ablation summary, baseline runtimes/errors.
- Save figures to `figures/final/` with descriptive names; keep generation scripts in `notebooks/` or `src/plots/`.

## Writing tasks
- Methods: data generation ranges, normalization, architecture specifics, loss weights, optimizer schedule, adaptive sampling steps, Greek computation via autograd.
- Evaluation: metric definitions, test and OOD setup, targets vs achieved, seeds, and any confidence intervals.
- Discussion: failure modes (ATM/short-dated Gamma noise, OOD volatility drift), mitigations attempted, limitations vs prior work.
- Refresh abstract/conclusion to reflect final claims; update related work and BibTeX.
- Consistency pass: units/notation, axis labels in figures, NeurIPS style length check (6–8 pages).

## Reproducibility
- Finalize scripts/notebooks for data synthesis, training, evaluation, and plotting; document run commands.
- Fix random seeds; capture environment versions; ensure `results/` holds metrics CSVs and figure assets.
- Validate `requirements.txt` matches used packages; note GPU/CPU assumptions.

## Timeline (suggested)
- Nov 4–17: core experiments, ablations, stability, baselines.
- Nov 18–24: finalize figures and tables; lock metrics.
- Nov 25–29: writing pass, citations, formatting; compile PDF draft.
- Nov 30: final polish and submission packaging.
