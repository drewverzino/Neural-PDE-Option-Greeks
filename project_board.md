# 🧮 CS 4644 / 7643 — Greeks Estimation via Physics-Informed Neural Networks

**Course:** Deep Learning (Fall 2025)  
**Project Weight:** 36% of total course grade  
**Team Members:** Drew Verzino, Rahul Rajesh, Aditya Deb, Navin Senthil  
**Version:** v2.0 — Integrated Course-Aligned Plan  

---

## 🎓 Project Purpose

The project allows students to:
1. Gain practical experience **implementing deep models**, and  
2. Apply deep learning to **self-selected problems of interest**.

Effort level: at least **1.5× a homework assignment per team member** (2–4 people per group).  
Deliverables are cumulative and represent **36% of the course grade**.

---

## 🧾 Grading Breakdown

| Deliverable | Weight | Due Date | Description |
|:--|:--:|:--:|:--|
| 🧠 **Proposal** | 1 % | **Oct 1 2025** | Short project concept summary |
| 🧩 **Milestone Report** | 10 % | **Nov 3 2025** | 4-page CVPR-style progress check |
| 📊 **Final Report** | 20 % | **Nov 30 2025** | 6–8 page CVPR/NeurIPS-style paper |
| 🖼️ **Poster Session** | 5 % | **Dec 1 2025** | Klaus Atrium presentation |

> ⚠️ **Formatting Requirements:**  
> - Use the official **CVPR / NeurIPS LaTeX template** (provided).  
> - Standard fonts, margins, and sizes — no edits for spacing.  
> - **Final report (6–8 pages)** must be self-contained.  
> - You may link to supplementary code or materials, but evaluation focuses on the PDF.

---

## 🎯 Project Summary

**Title:** *Greeks Estimation via Physics-Informed Neural Networks*  

Accurate estimation of option Greeks is critical for financial risk management. Traditional methods (finite differences, Monte Carlo) are noisy or computationally expensive.  
This project integrates the **Black-Scholes PDE** into a neural network’s loss function, training a single PINN that generalizes across strike and volatility regimes while producing smooth, stable Greeks via automatic differentiation.

---

## 🗓️ Phase 1 — Foundations & Milestone (Oct 13 → Nov 3)

**Goal:** Implement full data + model pipeline and submit Milestone Report.

### Week 1 (Oct 13 – Oct 20): Environment + Data Pipeline
- [x] **Repo Setup** – `src/`, `data/`, `notebooks/`, `figures/`, `results/` **Owner:** Team
- [x] **Install Env** – PyTorch 2.x, NumPy, Matplotlib, SciPy, pandas **Owner:** Team
- [x] **Black-Scholes Utilities** – Implement `BS_price()` + analytic Δ Γ Θ ν ρ **Owner:** Team
- [x] **Synthetic Dataset** – Generate 1 M train / 100 k val / test triplets (S, t, σ) **Owner:** Team
- [x] **Preprocessing** – x = ln(S/K), τ = T–t, normalize [–1, 1] **Owner:** Team
- [x] **Visualization** – Plot price, Greek surfaces, dataset sanity charts **Owner:** Team
- [x] ✅ **Checkpoint 1:** Dataset + plots verified **Owner:** 

### Week 2 (Oct 21 – Oct 27): Baselines + Model Scaffolding
- [x] **Finite-Diff Baseline** – ε-shift Δ Γ; compare vs analytic **Owner:** Team
- [x] **Monte Carlo Baseline** – GBM paths + pathwise Δ **Owner:** Team
- [x] **PINN Model** – 5×128 ResNet layers + ReLU **Owner:** Team
- [x] **Loss Functions** – `L_price`, `L_PDE`, `L_boundary`, Sobolev λ = 0.01 **Owner:** Team
- [x] **Training Loop + Logging** – Adam, warm-up, grad clip = 1.0 **Owner:** Team
- [x] ✅ **Checkpoint 2:** Model runs 1 epoch cleanly **Owner:** Team

### Week 3 (Oct 28 – Nov 3): Training + Milestone Report
- [x] **Base Training (50 epochs)** – Log loss curves (L_price, L_PDE) **Owner:** Team
- [x] **Full run evaluation notebook** – Capture proposal-ready training + OOS artefacts **Owner:** Team
- [x] **Diagnostics + Visuals** – PDE residual heatmap, Δ/Γ surfaces **Owner:** Team
- [x] **Adaptive Sampling Prototype** – Top 10 k error resampling **Owner:** Team
- [x] **RMSE Comparison (OOS CLI)** – Automate PINN vs FD & MC baselines **Owner:** Team
- [ ] **Write Milestone Report** – Intro, Methods, Prelim Results, Next Steps **Owner:** Team
- [ ] ✅ **Submit Milestone Report → Nov 3** **Owner:** Team

---

## 🧠 Phase 2 — Core Results & Experiments (Nov 4 → Nov 24)

**Goal:** Achieve quantitative + qualitative results for final paper.

### Week 4 (Nov 4 – Nov 10): Refinement + Hyperparameter Tuning
- [ ] Sweep λ ∈ {0.001, 0.01, 0.1} **Owner:** Team
- [ ] Tune batch size and learning rate schedule **Owner:** Team
- [ ] Validate training stability (3 seeds) **Owner:** Team
- [ ] Profile runtime and GPU memory **Owner:** Team
- [ ] ✅ Stable training configuration locked in **Owner:** Team

### Week 5 (Nov 11 – Nov 17): Quantitative Evaluation
- [ ] Compute RMSE for V, Δ, Γ, Θ, ν vs analytic BS **Owner:** Team
- [ ] Evaluate smoothness (Total Variation of Γ) **Owner:** Team
- [ ] Benchmark runtime (< 1 ms target) **Owner:** Team
- [ ] Ablation study (3 / 5 / 7 layers) **Owner:** Team
- [ ] ✅ Results tables + ablation plots finalized **Owner:** Team

### Week 6 (Nov 18 – Nov 24): Visualization + Interpretability
- [ ] Δ, Γ, ν surfaces vs S, σ for multiple τ **Owner:** Team
- [ ] PDE residual heatmaps + failure regions **Owner:** Team
- [ ] Smoothness progression plots **Owner:** Team
- [ ] Export all figures → `figures/final/` **Owner:** Team
- [ ] ✅ **Checkpoint 3:** All experiments and plots ready **Owner:** Team

---

## 🧾 Phase 3 — Final Report & Poster (Nov 25 → Dec 1)

**Goal:** Produce final paper and presentation materials.

### Week 7 (Nov 25 – Nov 30): Final Report
- [ ] Write Discussion + Conclusion (interpret results, limitations) **Owner:** Team
- [ ] Integrate figures + tables with captions **Owner:** Team
- [ ] Verify citations and BibTeX entries **Owner:** Team
- [ ] Proofread and compile 8-page CVPR PDF **Owner:** Team
- [ ] ✅ **Final Report Due → Nov 30** **Owner:** Team

### Week 8 (Dec 1): Poster Session
- [ ] Design poster layout (abstract, model, results, figures) **Owner:** Team
- [ ] Prepare 2–3 min presentation script **Owner:** Team
- [ ] Print poster and check layout clarity **Owner:** Team
- [ ] ✅ **Poster Session → Dec 1 (Klaus Atrium)** **Owner:** Team

---

## 📦 Major Deliverables Summary

| Date | Deliverable | Format | Weight | Status |
|:--:|:--|:--|:--:|:--:|
| Oct 1 | Proposal | 1 pg summary | 1 % | ✅ |
| Nov 3 | Milestone Report | 4-pg CVPR PDF | 10 % | ☐ |
| Nov 30 | Final Report | 6–8 pg CVPR PDF | 20 % | ☐ |
| Dec 1 | Poster Session | Printed poster | 5 % | ☐ |

---

## ⚠️ Risk & Contingency Plan

| Risk | Impact | Mitigation |
|:--|:--|:--|
| **Training instability** | Model fails to satisfy PDE | Grad clipping, residual layers, lower LR |
| **Noisy higher-order Greeks** | Γ, ν unstable or unsmooth | Sobolev penalty λ, multi-task Greek loss |
| **Compute limits** | Long train times | Colab A100 + checkpoint resume |
| **Time constraints** | Miss milestone | Focus on baseline + partial results first |
| **Formatting issues** | Grade penalty | Validate CVPR template + page count early |

---

## 🗂️ Optional Kanban View

| 🧩 To Do | ⚙️ In Progress | 📊 Done |
|:--|:--|:--|
| Milestone report draft | Expanded RMSE benchmarking | Repo setup |
| Poster design | Hyperparameter sweep plan | Black-Scholes utilities |
| Runtime benchmarking |  | Baselines & loss implementations |
| Final report outline |  | Preprocessing & adaptive sampling |
|  |  | Stress-test notebook |
|  |  | Full run evaluation notebook |
|  |  | OOS evaluation CLI & dashboards |
|  |  | End-to-end evaluation notebook |
|  |  | Training curves & logs |

---

**End of Project Plan**
