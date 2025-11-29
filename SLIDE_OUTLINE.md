# Detailed Slide Outline for Video Presentation

## Slide-by-Slide Content (10 Slides, 4:50 total)

---

### SLIDE 1: TITLE
**Duration: 0:15**

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   Greeks Estimation via Physics-Informed Neural Networks ║
║                                                           ║
║   Andrew Verzino  •  Rahul Rajesh  •  Aditya Deb         ║
║              •  Navin Senthil  •                          ║
║                                                           ║
║          Georgia Institute of Technology                  ║
║               CS 4644/7643 Fall 2025                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

[Background: Subtle financial chart or option price curve]
```

**Speaker:** Andrew
**Script:** Brief welcome + introduce team

---

### SLIDE 2: PROBLEM & MOTIVATION
**Duration: 0:35 (Cumulative: 0:50)**

**Title:** The Challenge: Real-Time Greeks Computation

**Left Panel (40%):**
```
📊 What are Greeks?
• Derivatives of option prices
• Essential for risk management
• Needed in real-time for trading

Example: Delta (Δ) = ∂V/∂S
         Gamma (Γ) = ∂²V/∂S²
```

**Right Panel (60%):**
```
╔═══════════════════════════════════════════════════╗
║ Method          │ Speed │ Accuracy │ Problem     ║
║─────────────────┼───────┼──────────┼─────────────║
║ Finite Diff     │ ✓     │ Biased   │ 2nd order  ║
║ Monte Carlo     │ ✗     │ ✓        │ Too slow   ║
║ Our PINN        │ ✓     │ ✓        │ ?          ║
╚═══════════════════════════════════════════════════╝

🎯 Research Question:
Can a single neural network compute accurate Greeks
across different volatility regimes instantly?
```

**Speaker:** Rahul
**Key point:** Traditional methods are either fast OR accurate, not both

---

### SLIDE 3: TECHNICAL APPROACH - ARCHITECTURE
**Duration: 0:45 (Cumulative: 1:35)**

**Title:** Physics-Informed Neural Network Architecture

**Main Visual (Center):**
```
Input: (S, t, σ) ──┐
                   │
Stock Price S ─────┤
Time t ────────────┼──► [Input Layer (128)]
Volatility σ ──────┘         │
                             │
                        ┌────▼─────┐
                        │ Residual │
                        │ Block 1  │ ×5 layers
                        │ + LayerN │
                        └────┬─────┘
                             │
                     [Output: Price V]
                             │
                    ┌────────▼─────────┐
                    │ Automatic Diff   │
                    │ (PyTorch Autograd)│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          Delta Δ       Gamma Γ        Vega ν
```

**Bottom Panel:**
```
🔑 Key Innovation: Volatility σ as input
   → No retraining needed for different market conditions!
```

**Speaker:** Aditya
**Key point:** Architecture enables direct Greek computation via autodiff

---

### SLIDE 4: PHYSICS-INFORMED LOSS FUNCTION
**Duration: 0:30 (Cumulative: 2:05)**

**Title:** Embedding Financial Theory into Training

**Top Panel (Loss Components):**
```
𝓛_total = 𝓛_price + 𝓛_PDE + 𝓛_boundary + 𝓛_smooth

𝓛_price:     Match Black-Scholes analytical prices
𝓛_PDE:       Satisfy Black-Scholes equation
𝓛_boundary:  Enforce payoff at expiration
𝓛_smooth:    Regularize for stable Greeks
```

**Center (PDE Equation):**
```
Black-Scholes PDE:

∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
  ↑       ↑        ↑         ↑        ↑
Theta   Sigma    Gamma     Delta   Discount
```

**Bottom (Visual):**
```
[Include: pde_residual_coarse.png heatmap]
Caption: PDE violations concentrate where options are most sensitive
```

**Speaker:** Navin
**Key point:** Physics constraint acts as powerful regularizer

---

### SLIDE 5: EXPERIMENTAL SETUP
**Duration: 0:20 (Cumulative: 2:25)**

**Title:** Evaluation Protocol

**Left Panel (50%):**
```
📊 Dataset
• 1M training samples
• Synthetic Black-Scholes
• S ∈ [20, 200]
• t ∈ [0.01, 1.99] years
• σ ∈ [0.05, 0.60] (train)
• σ ∈ [0.60, 0.65] (OOD test)

⚙️ Training
• 100 epochs
• Adam optimizer
• Adaptive sampling every 5 epochs
• Batch size: 4096
```

**Right Panel (50%):**
```
🎯 Three Hypotheses

H1: Volatility Generalization
    Can we achieve MAE_Δ < 0.05 and
    MAE_Γ < 0.10 across all σ?

H2: Physics Improves OOD
    Does PDE loss improve generalization
    by ≥15% vs supervised baseline?

H3: Adaptive Sampling Helps
    Does it reduce training time by ≥10%?

📐 Baselines
• Finite Differences (ε = 0.01)
• Monte Carlo (50k paths)
• Supervised MLP (no physics)
```

**Speaker:** Andrew
**Key point:** Rigorous experimental design with testable hypotheses

---

### SLIDE 6: MAIN RESULTS
**Duration: 0:45 (Cumulative: 3:10)**

**Title:** Experimental Results: All Hypotheses Confirmed ✓

**Top Table (Main Results):**
```
╔══════════════════════════════════════════════════════════════╗
║ Metric      │ PINN    │ Monte Carlo │ Target  │ Status      ║
║─────────────┼─────────┼─────────────┼─────────┼─────────────║
║ Delta MAE   │ 0.0085  │ 0.0010      │ < 0.05  │ ✓ Exceeded  ║
║ Gamma MAE   │ 0.00053 │ 0.00042     │ < 0.10  │ ✓ Exceeded  ║
║ Theta MAE   │ 0.393   │ 11.56       │ < 0.05  │ ⚠ Future    ║
║ Vega MAE    │ 1.916   │ 0.269       │ < 0.05  │ ⚠ Future    ║
║ Gamma TV    │ 0.98    │ 1.00        │ < 2.0   │ ✓ Smooth!   ║
╚══════════════════════════════════════════════════════════════╝
```

**Bottom Panels (3 columns):**
```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  H1: Volatility ✓   │  H2: Physics ✓      │  H3: Adaptive ✓     │
│                     │                     │                     │
│  Delta/Gamma meet   │  18% OOD boost      │  17% faster         │
│  targets across     │  vs supervised      │  convergence        │
│  σ ∈ [0.05, 0.60]   │  baseline           │                     │
│                     │                     │                     │
│  [Bar chart]        │  [Bar chart]        │  [Line chart]       │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

**Speaker:** Rahul
**Key point:** Excellent first-order Greeks; higher-order needs work

---

### SLIDE 7: VISUALIZATIONS
**Duration: 0:25 (Cumulative: 3:35)**

**Title:** Learned Surfaces: Smooth and Accurate

**Layout: 3-Panel Wide**
```
┌───────────────────┬───────────────────┬───────────────────┐
│  Price Surface    │  Greek Surfaces   │  PDE Residual     │
│                   │                   │                   │
│ [pinn_surface_    │ [pinn_delta_      │ [pde_residual_    │
│  3d.png]          │  surface.png]     │  coarse.png]      │
│                   │                   │                   │
│                   │ [pinn_gamma_      │                   │
│                   │  surface.png]     │                   │
│                   │                   │                   │
│ Smooth across     │ Match analytical  │ High residuals    │
│ (S, σ) space      │ solutions         │ near strike K     │
└───────────────────┴───────────────────┴───────────────────┘
```

**Bottom Caption:**
```
✓ Smooth interpolation ✓ Accurate derivatives ✓ Physics violations localized
```

**Speaker:** Aditya
**Key point:** Visual evidence of learning quality

---

### SLIDE 8: ABLATION STUDIES
**Duration: 0:25 (Cumulative: 4:00)**

**Title:** What Drives Performance?

**3-Column Layout:**
```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ Sobolev Weight λ     │ Adaptive Sampling    │ Physics-Informed     │
│                      │                      │                      │
│ λ    │ Final Loss    │ Method   │ Val RMSE  │ Model    │ OOD RMSE  │
│──────┼──────────     │──────────┼──────     │──────────┼──────     │
│ 0.001│ 2.006         │ Uniform  │ 9.95      │ Superv.  │ 0.62      │
│ 0.01 │ 2.168         │ Adaptive │ 8.79      │ PINN     │ 0.51      │
│ 0.1  │ 1.864 ✓       │          │           │          │           │
│      │               │ 12% ↓    │           │ 18% ↓    │           │
│ [Bar chart]          │ [Line chart]         │ [Bar chart]          │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

**Bottom Insight:**
```
💡 Key Insight: Physics regularization enables extrapolation beyond training data
```

**Speaker:** Navin
**Key point:** Design choices validated through ablations

---

### SLIDE 9: LIMITATIONS & FUTURE WORK
**Duration: 0:25 (Cumulative: 4:25)**

**Title:** Limitations and Next Steps

**Left Panel (40% - Red/Orange for caution):**
```
⚠️ Current Limitations

1. Higher-Order Greeks
   • Theta, Vega, Rho: 8-40× above targets
   • Need explicit supervision

2. Short Maturity
   • 2-3× errors near expiration
   • PDE becomes stiff

3. Extreme Moneyness
   • Larger relative errors
   • S ≫ K or S ≪ K
```

**Right Panel (60% - Green/Blue for opportunity):**
```
🚀 Future Directions

Technical Improvements
✓ Greek-specific loss terms
  𝓛_Greek = Σ (G_model - G_true)²

✓ Multi-task learning
  Separate heads for each Greek

✓ Time-dependent weighting
  Higher weight near expiration

Extensions
✓ American options (early exercise)
✓ Stochastic volatility (Heston, SABR)
✓ Multi-asset baskets
✓ Hybrid PINN + traditional methods
```

**Speaker:** Andrew
**Key point:** Honest about limitations, concrete future work

---

### SLIDE 10: CONCLUSIONS & IMPACT
**Duration: 0:25 (Cumulative: 4:50)**

**Title:** Summary: Fast, Accurate, Theory-Respecting Greeks

**Top Panel (Contributions):**
```
🎯 Key Contributions

✓ Single-Model Generalization
  One PINN across σ ∈ [0.05, 0.60] with <10% OOD degradation

✓ Physics-Informed Regularization
  18% improvement in extrapolation vs pure data-driven approach

✓ Adaptive Sampling Efficiency
  17% faster training by focusing on high-residual regions

✓ Hedge-Ready Smoothness
  Gamma TV ratio 0.98 → suitable for practical trading
```

**Bottom Panel (Impact):**
```
💼 Real-World Impact

Traditional: Hours of Monte Carlo simulation for scenario analysis
   PINN:     Instant evaluation across thousands of contracts

Who Benefits?
• Market makers: Real-time hedging decisions
• Risk managers: Rapid portfolio sensitivity analysis
• Quant traders: Fast strategy backtesting

📊 Tradeoff: Excellent first-order Greeks, room for improvement on higher-order
```

**Footer:**
```
Thank you! Questions?

Andrew Verzino • Rahul Rajesh • Aditya Deb • Navin Senthil
Georgia Institute of Technology
```

**Speaker:** Rahul
**Key point:** Strong conclusion emphasizing practical value

---

## Color Scheme Recommendations

```
Background:     White or very light gray (#F8F9FA)
Main Text:      Dark gray/black (#212529)
Headings:       Deep blue (#0056b3)
Success/Good:   Green (#28a745)
Warning:        Orange (#fd7e14)
Caution:        Red (#dc3545)
Highlights:     Gold/yellow background (#fff3cd)
Code/Math:      Monospace, light gray box (#e9ecef)
```

---

## Font Recommendations

```
Titles:         32-36pt, Bold, Sans-serif (Arial, Helvetica, Calibri)
Headings:       24-28pt, Bold
Body Text:      20-24pt, Regular
Captions:       16-18pt, Italic
Code/Equations: 18-20pt, Monospace (Consolas, Courier New)
```

---

## Animation Suggestions (Optional)

**Slide 2:** Fade in comparison table row by row
**Slide 3:** Build architecture diagram bottom-up
**Slide 6:** Highlight "✓ Exceeded" cells in green sequentially
**Slide 7:** Fade in surfaces left to right
**Slide 10:** Fade in contribution bullets one by one

**Warning:** Keep animations minimal—they eat into your 5-minute budget!

---

## Export Settings

**PowerPoint/Keynote:**
- Export as PDF (for backup)
- Embed all fonts
- Test on different screen (ensure readability)

**Recording:**
- 1920x1080 resolution minimum
- 30 fps
- MP4 format (H.264 codec for compatibility)
- Audio: 44.1kHz, 16-bit minimum

---

## Final Timing Breakdown Check

```
Slide 1:  0:15  (0:15 total)
Slide 2:  0:35  (0:50 total)
Slide 3:  0:45  (1:35 total)
Slide 4:  0:30  (2:05 total)
Slide 5:  0:20  (2:25 total)
Slide 6:  0:45  (3:10 total)
Slide 7:  0:25  (3:35 total)
Slide 8:  0:25  (4:00 total)
Slide 9:  0:25  (4:25 total)
Slide 10: 0:25  (4:50 total)
────────────────────────────
TOTAL:    4:50  (10 sec buffer! ✓)
```

This leaves you with a 10-second safety buffer before the 5:00 hard limit.
