# Greeks Estimation via Physics-Informed Neural Networks — Milestone Report

**Andrew Verzino** (averzino3@gatech.edu), **Rahul Rajesh** (rrajesh34@gatech.edu), **Aditya Deb** (adeb40@gatech.edu), **Navin Senthil** (nsenthil7@gatech.edu)  
Georgia Institute of Technology

---

## Abstract

Accurate estimation of option Greeks—sensitivities of an option’s value to underlying drivers—is essential for hedging and risk management. Classical approaches such as finite differences and Monte Carlo either introduce bias/variance issues or are too slow for real-time use. We propose a Physics-Informed Neural Network (PINN) that embeds the Black–Scholes PDE in the training loss and treats volatility $\sigma$ as an explicit input alongside spot price $S$ and time $t$. This allows a single trained model to generalize across volatility regimes and compute all major Greeks, including Vega $\partial V / \partial \sigma$, using automatic differentiation. We describe our implementation status, baselines, boundary-condition handling, and preliminary training behaviour.

---

## 1 Introduction

### 1.1 Problem Statement and Motivation

Greeks describe how an option’s value $V$ changes with respect to underlying parameters. Delta ($\Delta = \partial V / \partial S$) measures sensitivity to the underlying price $S$; gamma ($\Gamma = \partial^2 V / \partial S^2$) measures curvature; theta ($\Theta = -\partial V / \partial t$) measures time decay; vega ($\nu = \partial V / \partial \sigma$) measures volatility sensitivity; and rho ($\rho = \partial V / \partial r$) measures interest-rate sensitivity. Fast and accurate computation across many strikes and maturities is essential for risk management, but traditional methods like finite differences or Monte Carlo simulations are often too slow or biased—especially for higher-order Greeks. PINNs address these issues by embedding the Black–Scholes PDE and boundary conditions into the loss function, enabling fast Greek evaluation via automatic differentiation. However, most prior models assume fixed volatility, limiting their ability to compute vega or handle varying volatility conditions.

### 1.2 Approach and Contributions

We extend the PINN framework by treating volatility $\sigma$ as an explicit input dimension alongside $S$ and $t$. The model learns $V_\theta(S, t, \sigma)$ over a three-dimensional domain and delivers:

1. **Cross-regime generalization:** a single trained model supports multiple volatility scenarios.
2. **Direct vega computation:** $\sigma$ appears in the input, so $\nu = \partial V_\theta / \partial \sigma$ is available via autograd.
3. **Smooth Greek surfaces:** PDE residual and boundary-condition losses encourage globally consistent, smooth Greeks across $(S, t, \sigma)$.

**Input/output specification.** We transform raw inputs to log-moneyness $x = \ln(S/K)$ and time-to-maturity $\tau = T - t$, scale $(x, \tau, \sigma)$ to $[-1, 1]$, and predict prices $V_\theta$. Greeks follow from automatic differentiation.

**Training conditions.** Our current architecture is a 5-layer residual MLP (128 hidden units, $\tanh$ activations). The composite loss sums: (i) supervised MSE to analytic Black–Scholes prices, (ii) Black–Scholes PDE residual, (iii) boundary/terminal penalties (including the payoff), and (iv) Sobolev-style smoothness on $\partial_S V_\theta$ to stabilise gamma. Optimisation uses Adam with warmup ($10^{-5} \to 5 \times 10^{-4}$ over 500 steps), gradient clipping at 1.0, and an adaptive resampling loop that focuses on high PDE residual.

---

## 2 Related Work

### 2.1 Classical Methods

Finite differences approximate Greeks by repricing under small input perturbations but can be biased and unstable for higher-order derivatives [Haugh, 2017]. Monte Carlo methods (e.g., pathwise, likelihood ratio) yield unbiased estimates but require strong smoothness assumptions and exhibit high variance, especially for vega, while demanding large numbers of GBM paths [Glasserman, 2004; Jain et al., 2019].

### 2.2 Physics-Informed Neural Networks

PINNs embed governing PDEs and boundary conditions into neural training objectives, reducing data needs via physics-based regularisation [Raissi et al., 2019]. Tanios [2021] applied PINNs to high-dimensional option pricing, matching analytic benchmarks and producing Greeks through automatic differentiation. More recent work improves stability via residual backbones, LayerNorm, and adaptive sampling [Gao et al., 2025]. Our approach adds explicit boundary penalties to enforce terminal payoff and no-arbitrage conditions.

---

## 3 Problem Decomposition and Technical Approach

### 3.1 Research Hypotheses

- **H1 (Volatility generalization):** A single PINN with $\sigma$ as an input can achieve $\text{MAE}_\Delta < 0.05$ and $\text{MAE}_\Gamma < 0.10$ across $\sigma \in [0.05, 0.6]$, rivaling fixed-$\sigma$ PINNs.
- **H2 (Physics-informed regularisation):** Adding $\mathcal{L}_{\text{PDE}}$ improves OOD RMSE by ≥15% versus a price-only baseline on unseen $(S, t, \sigma)$.
- **H3 (Adaptive sampling efficiency):** Resampling high-residual regions cuts epochs required to reach $\text{RMSE}_\text{price} < 0.01$ by ≥10% versus uniform sampling.

### 3.2 Architecture and Preprocessing

**Network.** A residual MLP maps $\mathbb{R}^3 \to \mathbb{R}$ using five `TanhBlock`s (two fully connected layers with $\tanh$, residual connection, LayerNorm) and Xavier initialisation.

**Normalisation.** Raw $(S, t, \sigma)$ are transformed to $x = \log(S/K)$, $\tau = T - t$, then scaled to $[-1, 1]$ feature-wise.

### 3.3 Physics-Informed Loss

Total objective:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{price}} + \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{bc}} + \mathcal{L}_{\text{reg}}.
$$

- **Price fit:** $\mathcal{L}_{\text{price}} = \mathbb{E}[(V_\theta - V_{\text{BS}})^2]$.
- **PDE residual:** $\mathcal{L}_{\text{PDE}} = \mathbb{E}\big[(\partial_t V_\theta + \tfrac{1}{2}\sigma^2 S^2 \partial_{SS} V_\theta + r S \partial_S V_\theta - r V_\theta)^2\big]$.
- **Boundary conditions:** enforce payoff at maturity, zero limit as $S \to 0$, and deep ITM behaviour $S - K e^{-r(T - t)}$.
- **Smoothness:** $\mathcal{L}_{\text{reg}} = \lambda \,\mathbb{E}[(\partial_S V_\theta)^2]$ with $\lambda = 0.01$.

### 3.4 Training Protocol

- **Optimiser:** Adam with warmup, gradient clipping (norm ≤1.0).
- **Adaptive sampling:** every 5 epochs, evaluate residual on ~50k points, pick top 10k, jitter within domain bounds, refresh boundary samples, and augment the dataset.
- **Greek computation:** $\Delta = \partial_S V_\theta$, $\Gamma = \partial^2_{SS} V_\theta$, $\Theta = -\partial_t V_\theta$, $\nu = \partial_\sigma V_\theta$, $\rho = \tau (S \Delta - V_\theta)$.

### 3.5 Evaluation Metrics

- **Accuracy:** price RMSE and Greek MAE vs. analytic Black–Scholes on a held-out 20k test set. Targets: RMSE < 0.01; first-order MAE < 0.05; gamma MAE < 0.10.
- **Smoothness:** total variation of $\Gamma$ over $(S, \sigma)$ at fixed $t$, requiring $\text{TV}(\Gamma_\theta) < 2 \times \text{TV}(\Gamma_{\text{BS}})$.

---

## 4 Baseline Methods and Preliminary Results

### 4.1 Experimental Setup

- **Dataset:** $S \sim \mathcal{U}(20, 200)$, $t \sim \mathcal{U}(0.01, 1.99)$, $\sigma \sim \mathcal{U}(0.05, 0.6)$ with analytic prices/Greeks; 200k/40k/40k train/val/test (milestone-scale). Full runs use 1M/100k/100k.
- **Baselines:** analytic Black–Scholes (ground truth), central finite differences (epsilons 0.01/0.001), Monte Carlo pathwise with 50k GBM paths.

### 4.2 Baseline Performance

| Greek | Finite-difference MAE | Monte Carlo MAE | PINN MAE (initial) |
| --- | ---: | ---: | ---: |
| $\Delta$ | $4.61 \times 10^{-9}$ | 0.0014 | 0.1370 |
| $\Gamma$ | $4.60 \times 10^{-10}$ | 0.0006 | 0.0064 |
| $\Theta$ | 11.5625 | 11.5608 | 2.5382 |
| $\nu$ | $6.51 \times 10^{-5}$ | 0.3832 | 13.1704 |
| $\rho$ | $4.79 \times 10^{-7}$ | 0.0900 | 15.9082 |

Finite differences excel on first-order Greeks but amplify noise on $\Gamma$. Monte Carlo reduces bias but suffers high variance, especially for vega. Early PINN runs lag on supervised Greeks, motivating additional supervision and loss balancing.

### 4.3 Training Progress

- 50-epoch runs (batch 4096, warmup to $5 \times 10^{-4}$, adaptive sampling every 5 epochs) show steady PDE residual reduction and improved price fit.
- Residual heatmaps highlight ATM regions as hardest; adaptive resampling targets these areas.
- Loss curves reveal a brief warmup, then gradual convergence of all components.

---

## 5 Next Steps and Timeline

### 5.1 Remaining Work (Nov 4 – Nov 30)

- Sweep Sobolev penalty $\lambda \in \{0.001, 0.01, 0.1\}$ for gamma smoothness; test multiple seeds.
- Compute final test metrics (price RMSE, MAE for all Greeks, gamma TV); evaluate OOD performance on $\sigma \in [0.60, 0.65]$; ablate PINN vs. supervised-only networks.
- Generate 3D surfaces (price, delta, gamma, theta, vega) and error heatmaps; finalise hyperparameters; author final report.

### 5.2 Potential Challenges & Mitigation

- **Gamma instability:** increase $\lambda$, add explicit gamma supervision ($\partial^2_{SS} V_{\text{BS}}$), and explore Fourier-style features.
- **OOD generalisation:** extend training $\sigma$ to [0.05, 0.65], apply domain randomisation, and, if needed, scope claims to interpolation.

### 5.3 Success Criteria

- Minimum: price RMSE < 0.03, first-order MAE < 0.05.
- Target: price RMSE < 0.01, all Greek MAE < 0.05, gamma MAE < 0.10, <10% OOD degradation in volatility sweep, ≥15% improvement from $\mathcal{L}_{\text{PDE}}$.

---

## References

- Bae, H.-O., Kang, S., & Lee, M. (2024). *Option pricing and local volatility surface by physics-informed neural network*. Computational Economics, 64(5), 3143–3159.
- Ding, L., Lu, E., & Cheung, K. (2025). *Fast Derivative Valuation from Volatility Surfaces using Machine Learning*. arXiv:2505.22957.
- Gao, Q., Wang, Z., Zhang, R., & Wang, D. (2025). *Adaptive movement sampling physics-informed residual network (AM-PIRN) for solving nonlinear option pricing models*. arXiv:2504.03244.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Haugh, M. (2017). *Estimating the Greeks*. IEOR E4703 Lecture Notes, Columbia University.
- Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). *A nonparametric approach to pricing and hedging derivative securities via learning networks*. The Journal of Finance, 49(3), 851–889.
- Jain, S., Leitao, Á., & Oosterlee, C. W. (2019). *Rolling Adjoints: Fast Greeks along Monte Carlo scenarios for early-exercise options*. Journal of Computational Science, 33, 95–112.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs*. Journal of Computational Physics, 378, 686–707.
- Sirignano, J., & Spiliopoulos, K. (2018). *DGM: A deep learning algorithm for solving partial differential equations*. Journal of Computational Physics, 375, 1339–1364.
- Tanios, R. (2021). *Physics informed neural networks in computational finance: High dimensional forward & inverse option pricing*. Master’s thesis, ETH Zürich.
