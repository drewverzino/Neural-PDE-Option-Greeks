# CS 4644/7643: Greeks Estimation via Physics-Informed Neural Networks

## 📘 Overview

This repository contains the implementation for our **Deep Learning course project** at Georgia Tech — *Greeks Estimation via Physics-Informed Neural Networks (PINNs)*. The project explores how **deep neural networks** can be trained to **satisfy the Black–Scholes partial differential equation (PDE)** directly through the loss function, enabling accurate, efficient, and smooth computation of **option Greeks** (Δ, Γ, Θ, ν, ρ).

Traditional methods such as finite-difference and Monte Carlo simulations either produce noisy estimates or are computationally expensive, especially for higher-order Greeks. PINNs provide a powerful alternative by enforcing known physical (in this case, financial) laws during training.

---

## 🎯 Project Objectives

1. **Model Goal:** Develop a PINN that predicts option prices and Greeks simultaneously without retraining across volatility regimes.
2. **Key Innovation:** Include volatility (σ) as an explicit network input to generalize across different volatility surfaces.
3. **Evaluation:** Compare against classical baselines (Finite Difference, Monte Carlo, fixed-σ PINN) using metrics such as RMSE, total variation smoothness, and training stability.
4. **Application:** Improve the interpretability, efficiency, and numerical stability of deep learning models in quantitative finance.

---

## 📅 Deliverable Timeline

| Deliverable | Due Date | Description | Weight |
|:--|:--|:--|:--:|
| **Project Proposal** | Oct 1, 2025 | Define problem, related work, and plan | 1% |
| **Milestone Report** | Nov 3, 2025 | 4-page CVPR-style progress report | 10% |
| **Final Report** | Nov 30, 2025 | 6–8 page CVPR-style full paper | 20% |
| **Poster Session** | Dec 1, 2025 | In-person presentation (Klaus Atrium) | 5% |

---

## 🧱 Repository Structure

```
cs4644-pinn-greeks/
│
├── README.md                 # Project overview and setup guide
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignored files (datasets, logs, checkpoints)
│
├── src/                      # Source code
│   ├── data.py               # Data generation & preprocessing
│   ├── utils/
│   │   └── black_scholes.py  # Analytic pricing and Greek functions
│   ├── models/
│   │   └── pinn_model.py     # PINN architecture (Residual Network)
│   ├── losses.py             # Custom loss functions (L_price, L_PDE, etc.)
│   ├── train.py              # Training loop & adaptive sampling
│   └── eval.py               # Evaluation metrics and diagnostics
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── 01_data_visualization.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_training_diagnostics.ipynb
│
├── data/                     # Synthetic training and validation sets
│   ├── synthetic_train.npy
│   └── synthetic_val.npy
│
├── results/                  # Stored outputs, metrics, and tables
│   ├── baseline_fd.csv
│   ├── baseline_mc.csv
│   ├── pinn_results.csv
│   ├── ablation_study.csv
│   └── logs/                 # Training logs and W&B runs
│
├── figures/                  # Visualizations and plots
│   ├── data_exploration/
│   ├── training_curves/
│   ├── residual_heatmaps/
│   └── final_results/
│
└── reports/                  # Written deliverables
    ├── milestone/
    │   └── milestone_report.pdf
    ├── final/
    │   └── final_report.pdf
    └── poster/
        └── poster.pdf
```

---

## ⚙️ Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### `requirements.txt`
```
torch>=2.1
torchvision
numpy
scipy
pandas
matplotlib
seaborn
tqdm
wandb
jupyter
```
Optional (for LaTeX report compilation):
```
pylatexenc
```
---

## 💡 Implementation Details

### **1. Data Generation**
- Synthetic dataset generated from **Black–Scholes closed-form solution**.
- Inputs: stock price (S), time to maturity (t), volatility (σ).  
- Preprocessing: log-moneyness (x = ln(S/K)) and scaled time (τ = T–t).  
- Training/validation split: 1,000,000 train / 100,000 validation points.

### **2. Model Architecture**
- Fully connected **Residual Network** (5 layers × 128 neurons, ReLU).  
- Input: (S, t, σ) → Output: option price `Vθ(S, t, σ)`.  
- Residual connections stabilize gradient flow across layers.

### **3. Physics-Informed Loss**
\`\`\`math
L = L_{price} + α L_{PDE} + β L_{boundary} + λ‖∂_{SS}Vθ‖²
\`\`\`
- `L_price`: MSE between predicted and analytic prices.  
- `L_PDE`: PDE residual enforcing the Black–Scholes equation.  
- `L_boundary`: Terminal payoff condition \( V(T, S) = max(S-K, 0) \).  
- `λ‖∂SSV‖²`: Sobolev penalty for smoother Γ estimates.  

### **4. Baselines**
1. **Black–Scholes analytic** (ground truth)  
2. **Finite Difference** (ε-shift Δ, Γ)  
3. **Monte Carlo** (pathwise estimator)  
4. **Fixed-σ PINN** (retrained per volatility)

### **5. Evaluation Metrics**
| Metric | Description |
|:--|:--|
| **RMSE (V, Δ, Γ, Θ, ν)** | Error vs analytic Greeks |
| **Total Variation (Γ)** | Smoothness measure |
| **Runtime (ms)** | Inference efficiency |
| **Training Stability** | Convergence under adaptive sampling |

---

## 🚀 How to Run

### **Generate Data**
```bash
python src/data.py --n_train 1000000 --n_val 100000 --seed 42
```

### **Train PINN**
```bash
python src/train.py --epochs 100 --lr 1e-3 --batch_size 4096 --lambda_sobolev 0.01
```

### **Evaluate Baselines**
```bash
python src/eval.py --compare baselines
```

### **Plot Results**
```bash
python notebooks/03_training_diagnostics.ipynb
```

---

## 📊 Outputs & Visualizations

Expected visual outputs include:

- **Loss Curves:** L_price, L_PDE, L_boundary, total loss vs epoch  
- **PDE Residual Heatmaps:** visualize model satisfaction of PDE constraints  
- **Surface Plots:** Δ(S, σ), Γ(S, σ), ν(S, σ) across multiple expiries  
- **RMSE Tables:** model vs baselines (Black–Scholes, FD, MC)  
- **Ablation Charts:** performance vs network depth & Sobolev λ

---

## 🧩 Future Extensions

1. **Volatility Surface Calibration:** Extend from constant σ to implied volatility surfaces (SVI).  
2. **PINN for Exotic Options:** Apply to barrier or Asian options with path dependency.  
3. **Physics-Augmented Transformers:** Replace MLP with attention layers for higher flexibility.  
4. **Greeks Sensitivity Analysis:** Use automatic differentiation to visualize interdependence of Greeks.

---

## 👥 Team Members

| Name | Role | Responsibilities | Contact |
|:--|:--|:--|:--|
| **Drew Verzino** | Model / Training Lead | Model architecture, training scripts, report writing | |
| **Rahul Rajesh** | Math / PDE Lead | PINN loss design, PDE validation, theoretical background | |
| **Aditya Deb** | Data & Preprocessing Lead | Synthetic data generation, scaling, baseline integration | |
| **Navin Senthil** | Visualization / Reporting Lead | Diagnostic plots, LaTeX reports, poster design | |

---

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs.* Journal of Computational Physics, 378, 686–707.  
2. Tanios, R. (2021). *Physics Informed Neural Networks in Computational Finance: High Dimensional Forward & Inverse Option Pricing.* ETH Zürich Thesis.  
3. Bae, H.-O., Kang, S., & Lee, M. (2024). *Option Pricing and Local Volatility Surface by Physics-Informed Neural Network.* Computational Economics, 64(5), 3143–3159.  
4. du Plooy, R., & Venter, P. (2024). *Approximating Option Greeks in a Classical and Multi-Curve Framework Using Artificial Neural Networks.* Journal of Risk and Financial Management, 17(4):140.  
5. Gao, Q., Wang, Z., Zhang, R., & Wang, D. (2025). *Adaptive Movement Sampling Physics-Informed Residual Network (AM-PIRN) for Solving Nonlinear Option Pricing Models.* arXiv preprint arXiv:2504.03244.

---

## 🧾 License

This repository is for **academic use only** under the Georgia Tech CS 4644/7643 Deep Learning course. Redistribution or commercial use is prohibited without explicit permission from the course instructors.

---

**© 2025 — Georgia Institute of Technology | CS 4644/7643 Deep Learning Project**
