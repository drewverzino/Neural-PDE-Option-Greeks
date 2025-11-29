# Video Presentation Guide: Greeks Estimation via PINNs
## 5-Minute Presentation (Target: 4:45 to allow buffer)

---

## 🎬 Slide-by-Slide Script with Timing

### **SLIDE 1: Title (0:00 - 0:15)** — 15 seconds
**Speaker: Andrew**

> "Hi, I'm Andrew, and today our team will present 'Greeks Estimation via Physics-Informed Neural Networks.' We tackled the challenge of computing option price sensitivities—called Greeks—which are essential for risk management in financial markets but computationally expensive using traditional methods. I'm joined by Rahul, Aditya, and Navin."

**Visuals:**
- Title slide with team names
- Simple background: option price curve visual

---

### **SLIDE 2: Problem & Motivation (0:15 - 0:50)** — 35 seconds
**Speaker: Rahul**

> "Financial institutions need to evaluate thousands of option contracts in real-time to manage risk. The key metrics are Greeks—derivatives of option prices with respect to underlying parameters like stock price and volatility.
>
> Traditional methods have critical limitations: Finite differences are biased for higher-order Greeks, while Monte Carlo simulations are accurate but prohibitively slow, requiring tens of thousands of path simulations per estimate.
>
> Our goal: Can a single neural network learn to compute Greeks instantly across different market conditions?"

**Visuals:**
- **Left panel:** Trading desk needing real-time Greeks
- **Right panel:** Table comparing methods:
  ```
  Method          | Speed | Accuracy | Problem
  ----------------|-------|----------|----------
  Finite Diff     | Fast  | Biased   | 2nd order Greeks
  Monte Carlo     | Slow  | Unbiased | High variance
  Our PINN        | Fast  | Accurate | ?
  ```

**Time check: 0:50 / 4:45**

---

### **SLIDE 3: Technical Approach - Architecture (0:50 - 1:35)** — 45 seconds
**Speaker: Aditya**

> "Our approach uses a Physics-Informed Neural Network—a deep learning model that embeds financial theory directly into the training process.
>
> The architecture is a 5-layer residual network with 128 hidden units and layer normalization for stability. Crucially, we input stock price, time, AND volatility—unlike prior work that fixed volatility, forcing retraining for each market regime.
>
> The magic happens in our loss function, which has four components: First, supervised price matching against Black-Scholes. Second, a PDE residual term that enforces the Black-Scholes equation. Third, boundary conditions ensuring the model respects option payoffs at expiration. Fourth, a smoothness regularizer that keeps derivatives stable.
>
> After training, we compute all Greeks via automatic differentiation—no finite differences needed."

**Visuals:**
- **Architecture diagram:**
  ```
  Input: (S, t, σ) → [5 Residual Blocks] → Price V
                           ↓
                    Automatic Diff
                           ↓
              Greeks: Δ, Γ, Θ, ν, ρ
  ```
- **Loss function equation (simple):**
  ```
  L = L_price + L_PDE + L_boundary + L_smooth
  ```

**Time check: 1:35 / 4:45**

---

### **SLIDE 4: Key Innovation - Physics-Informed Loss (1:35 - 2:05)** — 30 seconds
**Speaker: Navin**

> "What makes this physics-informed? The PDE residual term forces the network to satisfy the Black-Scholes equation at every training point. This isn't just supervision—it's embedding 50 years of financial theory into the model.
>
> We also use adaptive sampling: every 5 epochs, we identify regions where the PDE is violated most—typically near-the-money, short-dated options—and add more training data there. This focuses learning where it matters most for practical trading."

**Visuals:**
- **PDE equation (simplified):**
  ```
  ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
  ```
- **Heatmap:** PDE residual showing high violations near strike price
  - Use your actual figure: `pde_residual_coarse.png`

**Time check: 2:05 / 4:45**

---

### **SLIDE 5: Experimental Setup (2:05 - 2:25)** — 20 seconds
**Speaker: Andrew**

> "We trained on 1 million synthetic Black-Scholes prices with volatility ranging from 5% to 60%. We tested three hypotheses: Can a single model generalize across volatilities? Does physics-informed training improve out-of-distribution performance? Does adaptive sampling accelerate convergence?
>
> We compare against finite differences and Monte Carlo with 50,000 paths."

**Visuals:**
- **Data specs (bullet points):**
  - 1M training samples
  - σ ∈ [0.05, 0.60] (training)
  - σ ∈ [0.60, 0.65] (OOD test)
  - 100 epochs, adaptive sampling every 5
- **Baselines:** FD, MC (50k paths), Supervised MLP

**Time check: 2:25 / 4:45**

---

### **SLIDE 6: Main Results (2:25 - 3:10)** — 45 seconds
**Speaker: Rahul**

> "Our results confirm all three hypotheses. For first-order Greeks, we achieve Delta mean absolute error of 0.0085 and Gamma error of 0.00053—both well below our targets and competitive with Monte Carlo.
>
> For out-of-distribution testing, we evaluated on unseen volatility regimes. The physics-informed PINN achieved 18% better generalization than a supervised baseline, and only 10% degradation from in-distribution performance. This shows the PDE constraint acts as a powerful regularizer.
>
> Adaptive sampling reduced training time by 17% compared to uniform sampling. The Gamma surface has total variation ratio of 0.98—meaning it's smooth enough for real hedging applications.
>
> However, higher-order Greeks like Theta and Vega remain challenging, with errors 8 to 40 times above targets. This suggests we need explicit Greek supervision in the loss function."

**Visuals:**
- **Table (simplified):**
  ```
  Metric      | PINN    | Monte Carlo | Target  | Status
  ------------|---------|-------------|---------|-------
  Delta MAE   | 0.0085  | 0.0010      | <0.05   | ✓
  Gamma MAE   | 0.00053 | 0.00042     | <0.10   | ✓
  Theta MAE   | 0.393   | 11.56       | <0.05   | ✗
  Vega MAE    | 1.916   | 0.269       | <0.05   | ✗
  ```
- **Bar chart:** OOD generalization (PINN vs Supervised: 18% improvement)

**Time check: 3:10 / 4:45**

---

### **SLIDE 7: Visualizations (3:10 - 3:35)** — 25 seconds
**Speaker: Aditya**

> "These surfaces show what the model learned. On the left, the price surface smoothly interpolates across stock prices and volatilities. The Delta and Gamma surfaces closely match the analytical solutions, with smooth gradients suitable for hedging. The PDE residual heatmap confirms violations concentrate near the strike price, exactly where options are most sensitive—this is why adaptive sampling helps."

**Visuals:**
- **3-panel figure:**
  1. Price surface (use `pinn_surface_3d.png`)
  2. Delta/Gamma surfaces side-by-side (use `pinn_delta_surface.png`, `pinn_gamma_surface.png`)
  3. PDE residual heatmap (use `pde_residual_coarse.png`)

**Time check: 3:35 / 4:45**

---

### **SLIDE 8: Ablations & Key Insights (3:35 - 4:00)** — 25 seconds
**Speaker: Navin**

> "Our ablation studies revealed three key insights. First, the Sobolev regularization weight lambda equals 0.1 gave best training loss convergence. Second, adaptive sampling not only accelerated training by 17% but also improved Gamma smoothness by 5%. Third, the physics-informed loss was critical—without the PDE term, OOD performance dropped by 18%. This confirms that embedding domain knowledge beats pure data-driven learning for extrapolation."

**Visuals:**
- **3 mini-charts:**
  1. Lambda sweep bar chart
  2. Training curves: adaptive vs uniform
  3. OOD performance: PINN vs supervised

**Time check: 4:00 / 4:45**

---

### **SLIDE 9: Limitations & Future Work (4:00 - 4:25)** — 25 seconds
**Speaker: Andrew**

> "Our main limitation is higher-order Greeks—Theta, Vega, and Rho need additional supervision. We also see 2-3x higher errors near expiration where the PDE becomes stiff.
>
> Future work should explore explicit Greek loss terms, multi-task learning with Greek-specific heads, and extending to American options and stochastic volatility models like Heston. Hybrid approaches combining PINNs for speed with finite differences for critical calculations could bridge the gap to production systems."

**Visuals:**
- **Left:** Failure modes (bullet points)
  - Higher-order Greeks: Θ, ν, ρ (8-40× above target)
  - Short maturity instability
  - Extreme moneyness errors
- **Right:** Future directions (bullet points)
  - Greek-specific loss terms
  - American options
  - Stochastic volatility (Heston, SABR)
  - Hybrid PINN + traditional methods

**Time check: 4:25 / 4:45**

---

### **SLIDE 10: Conclusions & Impact (4:25 - 4:50)** — 25 seconds
**Speaker: Rahul**

> "To conclude: We demonstrated that a single physics-informed neural network can generalize across volatility regimes without retraining, achieving excellent first-order Greek accuracy. The PDE constraint improves extrapolation by 18%, and adaptive sampling accelerates training by 17%.
>
> The impact? Market makers and risk managers could evaluate thousands of scenarios in real-time instead of waiting hours for Monte Carlo simulations. While challenges remain for higher-order Greeks, our results show PINNs are a promising direction for modern quantitative finance, balancing speed, accuracy, and theoretical consistency.
>
> Thank you!"

**Visuals:**
- **Key Contributions (4 bullet points):**
  - ✓ Single-model volatility generalization
  - ✓ 18% OOD improvement via physics
  - ✓ 17% training speedup via adaptive sampling
  - ✓ Smooth, hedge-ready Gamma surfaces
- **Impact statement:**
  - "Real-time scenario analysis for risk management"
- **Team photo or names**

**Time check: 4:50 / 5:00** ✅ **SAFE!**

---

## 📊 Speaking Role Distribution

| Speaker | Slides | Time | Role |
|---------|--------|------|------|
| Andrew  | 1, 5, 9 | ~1:05 | Architecture, setup, limitations |
| Rahul   | 2, 6, 10 | ~1:45 | Motivation, results, conclusion |
| Aditya  | 3, 7 | ~1:10 | Technical approach, visualizations |
| Navin   | 4, 8 | ~0:55 | Physics-informed loss, ablations |
| **Total** | **10 slides** | **4:55** | **All participate equally** |

---

## 🎯 Rubric Optimization Strategy

### Logistics (10 points)
- ✅ All 4 team members have substantial speaking roles (~1 minute each)
- ✅ Everyone introduces themselves or transitions smoothly

### Presentation Quality (20 points)
**Slides/Visuals (10 points):**
- Use high-contrast colors (dark text on light background)
- Large fonts (≥24pt for body text)
- Include your actual figures from the report (they're already publication-quality)
- Limit text: 3-5 bullet points per slide max

**Speaking/Pacing/Audio (10 points):**
- Practice with a timer—aim for 4:45 to leave 15-second buffer
- Record in a quiet room with good microphone
- Speak clearly at moderate pace (~140 words/min)
- Use transitions: "Now Aditya will explain our architecture..."

### Content (70 points)

**Problem & Motivation (15 points):**
- ✅ Clear problem statement: Greeks computation is slow/biased
- ✅ Importance: Real-time risk management needs
- ✅ Gap: Prior PINNs fix volatility, require retraining

**Technical Approach (20 points):**
- ✅ Architecture: 5-layer residual MLP with LayerNorm
- ✅ Key innovation: Volatility as input + physics-informed loss
- ✅ 4-component loss function explained
- ✅ Adaptive sampling strategy

**Results & Evaluation (20 points):**
- ✅ 3 baselines: FD, MC, supervised MLP
- ✅ Main results table with targets vs actuals
- ✅ H1, H2, H3 validation
- ✅ Limitations discussed (higher-order Greeks)

**Conclusions (15 points):**
- ✅ 4 key contributions listed
- ✅ Impact: Real-time scenario analysis
- ✅ Future work: 4 concrete directions

---

## 🎬 Production Tips

### Recording Setup
1. **Software:** Zoom (record locally), OBS Studio, or PowerPoint recording
2. **Format:** MP4, 1080p, 30fps minimum
3. **Audio:** Use headset mic or lapel mic (not laptop built-in)
4. **Lighting:** Face a window or use desk lamp (no backlighting)

### Recording Options

**Option A: Live Presentation (Recommended)**
- All 4 team members on Zoom
- Share screen with slides
- Record meeting
- More natural, shows teamwork

**Option B: Narrated Slides**
- Record each section separately
- Stitch together in video editor
- Cleaner, easier to re-record mistakes

### Quality Checklist
- [ ] Test recording: Can you read all slide text?
- [ ] Audio check: No echo, background noise, or clipping?
- [ ] Timing check: Under 5:00?
- [ ] All team members speak?
- [ ] Figures are visible and clear?
- [ ] Smooth transitions between speakers?

---

## ⚠️ Common Pitfalls to Avoid

### Time Management
- ❌ Don't go into related work details—keep it in introduction
- ❌ Don't explain every ablation—pick the most impactful
- ❌ Don't read equations—just explain what they do
- ✅ Practice 3 times minimum to nail timing

### Technical Depth
- ❌ Don't assume audience knows finance terms (explain Greeks clearly)
- ❌ Don't skip motivation (why should ML audience care?)
- ✅ Balance accessibility with technical rigor

### Visuals
- ❌ Don't use tiny fonts (<18pt)
- ❌ Don't cram multiple dense tables
- ✅ Use simple, high-contrast charts

---

## 🚀 Final Checklist Before Submission

### 48 Hours Before Deadline
- [ ] Script finalized and divided among team
- [ ] Slides created with all figures embedded
- [ ] Practice run #1 (likely over time)
- [ ] Cut content to fit under 5:00

### 24 Hours Before Deadline
- [ ] Practice run #2 (should be close to time)
- [ ] Record final version
- [ ] Watch recording together—check audio/video quality
- [ ] Re-record if needed

### 2 Hours Before Deadline
- [ ] Upload to Gradescope
- [ ] Verify video plays correctly
- [ ] Add all team members to group submission
- [ ] Watch uploaded video one more time

---

## 📈 Expected Score Breakdown

| Category | Expected | Why |
|----------|----------|-----|
| Logistics | 10/10 | All speak equally |
| Slides/Visuals | 10/10 | High-quality figures from report |
| Speaking | 9/10 | Clear, practiced delivery |
| Problem/Motivation | 14/15 | Strong, accessible intro |
| Technical Approach | 19/20 | All key innovations covered |
| Results | 19/20 | Tables, ablations, honest limitations |
| Conclusions | 14/15 | Clear contributions & impact |
| **Total** | **95/100** | **Excellent presentation** |

---

## 💡 Pro Tips for Standing Out

1. **Show personality:** Smile, make eye contact with camera, sound enthusiastic!
2. **Tell a story:** "Imagine you're a risk manager needing real-time Greeks..."
3. **Use animations:** Fade in bullet points, highlight key numbers
4. **End strong:** "While challenges remain, PINNs could transform quantitative finance"

Good luck! Your project is excellent—now showcase it well! 🎓
