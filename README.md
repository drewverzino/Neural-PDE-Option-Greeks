# Greeks Estimation via Physics-Informed Neural Networks

**CS 4644/7643: Deep Learning - Fall 2025**

**Team Members:** Drew Verzino, Rahul Rajesh, Aditya Deb, Navin Senthil

---

## Project Summary

Risk managers at financial institutions must continuously hedge derivatives portfolios by estimating option Greeks—sensitivities to price, volatility, time, and other parameters—across thousands of strikes and market conditions daily. Traditional methods like finite differences and Monte Carlo simulation are either noisy and biased or require 100+ milliseconds per evaluation with $10^5$+ paths. Physics-Informed Neural Networks (PINNs) offer fast inference (<1ms) by embedding the Black-Scholes PDE directly into training, but existing approaches fix volatility and must retrain entirely for each new parameter set.

We propose a unified PINN architecture that treats volatility $\sigma$ as an explicit network input alongside spot price $S$ and time $t$. This single architectural change enables: (1) zero-shot generalization across volatility regimes without retraining, (2) direct computation of Vega $(\frac{\partial V}{\partial\sigma})$ via automatic differentiation, which prior fixed $\sigma$ PINNs cannot access, and (3) smooth Greek surfaces across the entire $(S,t,\sigma)$ space from one trained model. We demonstrate this approach on European options, achieving analytic accuracy while reducing the computational cost of exploring volatility scenarios by 100x compared to retraining-based methods. We validate our method against Black-Scholes analytic formulas and benchmark against finite-difference and Monte Carlo estimators to verify both accuracy and computational efficiency.

---

## Related Work

### Definitions and Classical Methods

The Greeks are defined as partial derivatives of the option value with respect to underlying parameters. Investopedia provides accessible descriptions of Delta, Gamma, Theta, Vega, and Rho and explains their economic interpretation. Traditional approaches for computing Greeks include finite-difference approximations and Monte Carlo estimators. Haugh's lecture notes on Monte Carlo simulation describe three classical estimators: finite differences (which yield biased estimates), the pathwise estimator, and the likelihood-ratio method; the latter two can produce unbiased estimates under suitable smoothness conditions. The pathwise estimator interchanges differentiation and expectation to compute derivatives directly, but this technique requires that the payoff be sufficiently smooth and may fail for barrier or digital options. These methods often require a large number of simulated paths to reduce variance and thus remain computationally costly compared to surrogate models.

### Deep Learning for Option Pricing

Neural networks have been used to approximate option prices and Greeks. Bae, Kang and Lee demonstrated that feedforward networks can fit option prices and construct local volatility surfaces while matching analytic Greeks. Sirignano and Spiliopoulos introduced the Deep Galerkin Method (DGM) as a neural PDE solver, and Raissi, Perdikaris and Karniadakis proposed Physics-Informed Neural Networks (PINNs) for general PDEs. Tanios applied PINNs to high-dimensional Black-Scholes and Heston models and showed that Greeks such as Delta, Gamma and Theta can be extracted via automatic differentiation. Tanios also noted that Vega and Rho require treating volatility and interest rate as inputs or using derived relations. An MDPI article by Labuschagne and von Boetticher formalizes Greek definitions and describes their qualitative behaviour across different moneyness and maturities.

### Baseline Methods

The standard industry baseline for Greeks is finite differences: approximating $\frac{\partial V}{\partial S} \approx \frac{V(S+\epsilon) - V(S-\epsilon)}{2\epsilon}$, which requires $2n$ pricing calls for $n$ Greeks and introduces $O(\epsilon^2)$ truncation error plus numerical cancellation issues for small $\epsilon$. For stochastic models, the pathwise Monte Carlo estimator interchanges differentiation and expectation to produce unbiased Greeks but requires $10^4-10^6$ paths for acceptable variance and fails for discontinuous payoffs. Recent neural approaches learn option prices and match analytic Greeks within a fixed volatility regime, offering fast inference but requiring separate training for each volatility surface—a critical limitation for scenario analysis and stress testing.

### Limitations and Our Contribution

Existing PINN implementations for option pricing treat volatility as a fixed parameter embedded in the PDE coefficients, necessitating complete retraining to evaluate Greeks under different volatility assumptions. Tanios notes that Vega and Rho "cannot be derived from the PDE alone" without treating them as inputs or using auxiliary relations. An adaptive movement sampling approach (AM-PIRN) by Gao, Wang and co-authors proposed redistributing training points and using ResNet backbones to improve stability for nonlinear option-pricing PDEs. We address these limitations by: (1) extending the input space to $(S,t,\sigma)$, allowing direct Vega computation via $\partial_\sigma V_\theta$; (2) training once to achieve generalization across volatility regimes; and (3) incorporating adaptive sampling to improve convergence in high-curvature regions where Greeks are most difficult to estimate. This design enables real-time scenario analysis without retraining overhead.

---

## Proposed Method

We will design a fully connected residual neural network $\mathcal{N}\_{\theta}$ that takes inputs $(S,t,\sigma)$ and outputs the option value $V_\theta(S,t,\sigma)$. The loss function consists of three components: (i) a pricing loss enforcing that $V_\theta$ matches the analytic Black-Scholes prices on sampled training points; (ii) a PDE residual loss penalizing deviations from the Black-Scholes equation $$\partial_t V + \frac{1}{2}\sigma^2 S^2 \partial_{SS}V + r S \partial_S V - r V = 0$$; and (iii) a boundary and terminal loss enforcing the payoff $V(T,S,\sigma) = \max(S-K,0)$ and appropriate boundary conditions as $S \to 0$ and $S \to \infty$. To improve training stability, we will normalize inputs using log-moneyness $x = \log(S/K)$ and scaled time-to-maturity $\tau = T-t$, and incorporate residual connections to mitigate gradient pathologies. We will implement learning rate warmup (from $10^{-5}$ to $10^{-3}$ over 1000 steps) and gradient clipping (maximum norm of 1.0) to handle the multi-scale nature of the loss components. Adaptive sampling will concentrate collocation points in regions with large PDE residuals: after 50 epochs of initial training, we identify the 10,000 points with highest $|\mathcal{L}[V]|$ and add 10,000 new samples within a radius of 0.1 (in normalized coordinates) around these high-error regions, then continue training for 50 more epochs.

After training, Greeks will be obtained by differentiating the network output: $\Delta = \partial_S V_\theta$, $\Gamma = \partial_{SS} V_\theta$, $\Theta = -\partial_t V_\theta$ and $\nu = \partial_{\sigma} V_\theta$. Treating volatility as an input allows us to compute Vega directly via automatic differentiation, addressing the limitation noted by Tanios that Vega and Rho cannot be derived from the PDE alone when volatility is fixed. Should we extend the analysis to varying interest rates, we will either include $r$ as an additional input or exploit analytic relations between Greeks to compute Rho. To address potential noise in second-order derivatives, we will add a Sobolev regularization penalty $\lambda \int |\partial_{SS} V|^2$ with $\lambda = 0.01$ to encourage smooth Gamma surfaces.

### Rationale

Our approach should succeed for three reasons. First, the Black-Scholes PDE guarantees that $V(S,t,\sigma)$ is a smooth function across its domain, making it amenable to neural approximation—by including $\sigma$ in the input space, we exploit this smoothness rather than treating each volatility as a separate problem requiring retraining. Second, the PDE residual loss acts as a physics-informed regularizer that constrains the network to solutions respecting the known governing equation, which should improve generalization beyond the training distribution compared to pure supervised learning on prices alone. Third, automatic differentiation provides exact derivatives of the network (up to floating-point precision), avoiding the truncation error and numerical cancellation that plague finite-difference methods, particularly for higher-order Greeks like Gamma where errors compound.

### Evaluation

We will assess our method along three dimensions. For **accuracy**, we compare predicted Greeks against Black-Scholes analytic formulas using RMSE and MAPE on a held-out test set, targeting RMSE $< 0.01$ for option prices and $< 0.05$ for first-order Greeks ($\Delta$, $\Theta$, $\nu$). We compare against four **baselines**: (i) Black-Scholes analytic formulas (ground truth); (ii) finite-difference approximations with $\epsilon = 0.01$; (iii) Monte Carlo pathwise estimator with $10^5$ paths; and (iv) fixed $\sigma$ PINN retrained per volatility following Tanios. For **computational efficiency**, we measure inference time per evaluation (target: $<1$ms on CPU) and compare total time including training versus the retraining overhead required for multi-volatility scenarios with baseline (iv). To assess **smoothness**—critical for hedging applications—we compute the total variation of $\Gamma(S,\sigma)$ surfaces and require that it remains within $2\times$ the analytic Black-Scholes value, since noisy Greeks are unusable for risk management despite acceptable mean error. We will visualize Delta and Gamma surfaces across $(S,\sigma)$ at fixed maturity, PDE residual heatmaps before and after adaptive sampling, and Greek profiles at representative strikes to provide qualitative assessment of surface quality.

---

## Datasets/Environments

We will generate synthetic datasets using the Black-Scholes closed-form solution for European call options. The training set consists of 80,000 samples drawn via Sobol sequences to ensure low-discrepancy coverage of the parameter space: $S/K \in [0.4, 1.8]$, $\tau \in [0.01, 1.0]$ years, and $\sigma \in [0.05, 0.6]$. The test set contains 20,000 samples split into in-distribution (15,000 samples) and out-of-distribution (5,000 samples with $\sigma \in [0.6, 0.65]$) to measure generalization. We fix the risk-free rate $r = 0.05$ and strike $K = 100$ throughout.

During training, we will apply one round of adaptive refinement: after 50 epochs, identify the 10,000 points with highest PDE residual $|\mathcal{L}[V]|$, add 10,000 new points sampled uniformly within a radius of 0.1 (in normalized coordinates) around these high-error regions, and continue training for 50 more epochs. For each sampled $(S,t,\sigma)$, we will compute the Black-Scholes option price and analytic Greeks to serve as both training supervision and evaluation ground truth.

All experiments will be conducted on a single NVIDIA V100 GPU with 16GB memory. We estimate 2-4 hours for initial training and 30 minutes per adaptive refinement iteration, making the total workflow feasible within typical academic compute constraints. The network will be implemented in PyTorch.

---

## Potential Risks

### Risk 1: Training Instability

PINN loss functions combine terms with different scales (price errors $\sim O(10)$, PDE residuals $\sim O(1)$, derivative terms $\sim O(0.1)$), which can cause gradient pathologies and slow convergence.

**Mitigation:** We will implement: (1) learning rate warmup ($10^{-5} \to 10^{-3}$ over 1000 steps), (2) gradient clipping (max norm = 1.0), and (3) loss balancing using either fixed weights tuned via grid search or learned weights via uncertainty-based weighting. If convergence fails after 100 epochs (validation loss not decreasing), we will fall back to curriculum learning: train first on a narrow volatility range $\sigma \in [0.15, 0.25]$, then progressively expand to the full range.

### Risk 2: Noisy Second-Order Greeks

Gamma $= \partial^2 V/\partial S^2$ amplifies any roughness in the learned surface, potentially yielding unusable hedging parameters despite acceptable price RMSE.

**Mitigation:** We will add a Sobolev penalty $\lambda \int |\partial^2 V/\partial S^2|^2$ to the loss with $\lambda = 0.01$. We define success as total variation $\text{TV}(\Gamma) < 2\times$ the analytic Black-Scholes TV on the test set; if this is not achieved, we will increase $\lambda$ or explore spectral methods (Fourier features with lower frequency cutoffs to naturally smooth the learned function).

**Failure mode:** If Gamma RMSE $> 0.2$ or $\text{TV}(\Gamma) > 5\times$ analytic after all mitigations, we will document this limitation and restrict claims to first-order Greeks only.

### Risk 3: Limited Applicability

Our method is designed for vanilla European options under Black-Scholes. Extending to American options (requiring optimal stopping boundaries) or exotic payoffs (barriers, digitals) may require architectural modifications beyond this project's scope.

**Mitigation:** We will include one ablation study with a digital option payoff to identify specific failure modes (e.g., gradient instability near discontinuities). This will inform a "Future Work" discussion on how to adapt the method (e.g., using separate networks for smooth and discontinuous components).

### Success Criteria

We define three tiers of success:
- **Minimum viable:** MAPE $< 5\%$ for $V$, $\Delta$, $\Theta$ on in-distribution test data, matching finite-difference baseline accuracy.
- **Target:** MAPE $< 5\%$ maintained on out-of-distribution $\sigma$ with $<10\%$ degradation, plus 10x faster inference than Monte Carlo.
- **Stretch:** Gamma RMSE $< 0.1$ with smooth surfaces (TV $< 2\times$ analytic), enabling direct use for delta-gamma hedging.

---

## References

1. Labuschagne, C. C., & von Boetticher, S. (2017). Approximating Option Greeks in a Classical and Multi-Curve Framework Using Artificial Neural Networks. *Journal of Risk and Financial Management*.

2. Investopedia Editors. Options Trading Decoded: 5 Key "Greeks" You Must Understand. *Investopedia*. (Accessed 2025).

3. Bae, H.-O., Kang, S., & Lee, M. (2024). Option Pricing and Local Volatility Surface by Physics-Informed Neural Network. *Computational Economics*.

4. Ding, L., Lu, E., & Cheung, K. (2025). Fast Derivative Valuation from Volatility Surfaces using Machine Learning. arXiv preprint arXiv:2505.22957.

5. Tanios, R. (2021). Physics-Informed Neural Networks in Computational Finance: High Dimensional Forward & Inverse Option Pricing. Master's Thesis, ETH Zürich.

6. Gao, Q., Wang, Z., Zhang, R., & Wang, D. (2025). Adaptive Movement Sampling Physics-Informed Residual Network (AM-PIRN) for Solving Nonlinear Option Pricing Models. arXiv preprint arXiv:2504.03244.

7. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A deep learning framework for solving PDEs. *Journal of Computational Physics*.

8. Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. *Journal of Computational Physics*.

9. Haugh, M. (2017). Estimating the Greeks. Lecture notes, IEOR E4703: Monte-Carlo Simulation, Columbia University.

---

## Implementation Timeline (Tentative)

- **Weeks 1-2:** Data generation and baseline implementation
- **Weeks 3-4:** PINN architecture and basic training
- **Weeks 5-6:** Adaptive sampling and regularization experiments
- **Weeks 7-8:** Evaluation, comparison with baselines, and report writing

---

*Project for CS 4644/7643: Deep Learning, Fall 2025*
*Georgia Institute of Technology*
