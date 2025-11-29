# FULL VIDEO PRESENTATION SCRIPT
## Greeks Estimation via Physics-Informed Neural Networks
### CS 4644/7643 Deep Learning - Final Presentation
### Total Duration: 4 minutes 50 seconds

---

## 🎬 SLIDE 1: TITLE SLIDE
**Duration: 15 seconds | Cumulative: 0:15**
**Speaker: ANDREW**

---

**[ANDREW - On camera, welcoming tone]**

"Hi everyone, I'm Andrew Verzino. Today our team will present 'Greeks Estimation via Physics-Informed Neural Networks.' We're tackling the challenge of computing option price sensitivities—called Greeks—which are essential for risk management in financial markets but computationally expensive using traditional methods. I'm joined by my teammates Rahul Rajesh, Aditya Deb, and Navin Senthil from Georgia Tech. Rahul will start by explaining the problem."

**[Transition to Slide 2]**

---

## 📊 SLIDE 2: PROBLEM & MOTIVATION
**Duration: 35 seconds | Cumulative: 0:50**
**Speaker: RAHUL**

---

**[RAHUL - Clear, engaging delivery]**

"Thanks Andrew. Financial institutions need to evaluate thousands of option contracts in real-time to manage portfolio risk. The key metrics they care about are Greeks—these are derivatives of option prices with respect to underlying parameters. For example, Delta measures how the option price changes with the stock price, and Gamma measures the curvature of that relationship.

Traditional methods have critical limitations. Finite differences are fast but produce biased estimates, especially for second-order Greeks like Gamma. Monte Carlo simulations are more accurate but prohibitively slow—they require simulating tens of thousands of price paths for each estimate.

This brings us to our research question: Can a single neural network learn to compute accurate Greeks instantly across different market volatility conditions without requiring retraining? Aditya will now explain our technical approach."

**[Transition to Slide 3]**

---

## 🏗️ SLIDE 3: TECHNICAL APPROACH - ARCHITECTURE
**Duration: 45 seconds | Cumulative: 1:35**
**Speaker: ADITYA**

---

**[ADITYA - Confident, technical but accessible]**

"Our solution uses a Physics-Informed Neural Network—a deep learning model that embeds financial theory directly into the training process.

The architecture is a 5-layer residual network with 128 hidden units per layer and layer normalization for training stability. Here's the crucial innovation: we input stock price, time, AND volatility into the network. Unlike prior work that fixed volatility and required retraining for each market regime, our single model handles the entire volatility spectrum.

The magic happens in our loss function, which has four components. First, supervised price matching against the Black-Scholes analytical solution. Second, a PDE residual term that enforces the Black-Scholes partial differential equation. Third, boundary conditions ensuring the model respects option payoffs at expiration. And fourth, a Sobolev smoothness regularizer that keeps the derivatives stable for practical hedging.

After training, we compute all Greeks—Delta, Gamma, Theta, Vega, and Rho—via automatic differentiation. No finite differences needed. Navin will explain what makes this physics-informed."

**[Transition to Slide 4]**

---

## ⚙️ SLIDE 4: PHYSICS-INFORMED LOSS FUNCTION
**Duration: 30 seconds | Cumulative: 2:05**
**Speaker: NAVIN**

---

**[NAVIN - Emphasize the key insight]**

"What makes this physics-informed? The PDE residual term forces the network to satisfy the Black-Scholes equation at every training point. You can see the equation here—it relates how the option value changes with time, the second derivative with respect to stock price, and the drift and discount terms.

This isn't just supervision—it's embedding fifty years of financial theory into the model. The PDE acts as a powerful regularizer that constrains the solution space.

We also use adaptive sampling: every five epochs, we identify regions where the PDE is violated most—shown here in red on this heatmap. These are typically near-the-money, short-dated options. We add more training data in these high-residual regions, focusing learning where it matters most for practical trading. Andrew will now describe our experimental setup."

**[Transition to Slide 5]**

---

## 🔬 SLIDE 5: EXPERIMENTAL SETUP
**Duration: 20 seconds | Cumulative: 2:25**
**Speaker: ANDREW**

---

**[ANDREW - Clear, methodical]**

"We trained on one million synthetic Black-Scholes prices with volatility ranging from five percent to sixty percent. We held out one hundred thousand samples for testing, including an out-of-distribution test set with volatilities from sixty to sixty-five percent—beyond our training range.

We tested three hypotheses: Can a single model generalize across volatilities? Does physics-informed training improve out-of-distribution performance? And does adaptive sampling accelerate convergence?

For baselines, we compare against finite differences and Monte Carlo with fifty thousand paths. Rahul will now present our results."

**[Transition to Slide 6]**

---

## 📈 SLIDE 6: MAIN RESULTS
**Duration: 45 seconds | Cumulative: 3:10**
**Speaker: RAHUL**

---

**[RAHUL - Enthusiastic about results, honest about limitations]**

"Our results confirm all three hypotheses. Looking at this table, for first-order Greeks we achieve Delta mean absolute error of zero point zero zero eight five and Gamma error of zero point zero zero zero five three—both well below our target thresholds and competitive with Monte Carlo.

For hypothesis two, we evaluated on unseen volatility regimes. The physics-informed PINN achieved eighteen percent better generalization than a supervised baseline trained only on prices. This shows the PDE constraint acts as a powerful regularizer.

For hypothesis three, adaptive sampling reduced training time by seventeen percent compared to uniform sampling while improving final accuracy.

Our Gamma surface has a total variation ratio of zero point nine eight—meaning it's smooth enough for real hedging applications.

However, I want to be honest about our limitations. Higher-order Greeks like Theta and Vega remain challenging, with errors eight to forty times above our targets. This suggests we need to add explicit Greek supervision terms to the loss function in future work. Aditya will show what the model learned visually."

**[Transition to Slide 7]**

---

## 📊 SLIDE 7: VISUALIZATIONS
**Duration: 25 seconds | Cumulative: 3:35**
**Speaker: ADITYA**

---

**[ADITYA - Point to specific features in the figures]**

"These surfaces show what the model learned. On the left, the price surface smoothly interpolates across stock prices and volatilities—this is the parametric family of solutions we mentioned.

In the center, the Delta and Gamma surfaces closely match the analytical Black-Scholes solutions. Notice the smooth gradients—this is critical for hedging, where you need stable derivative estimates.

On the right, the PDE residual heatmap confirms that violations concentrate near the strike price at one hundred dollars, exactly where option curvature is highest. This validates why adaptive sampling helps—we're focusing compute where the PDE is hardest to satisfy. Navin will walk through our ablation studies."

**[Transition to Slide 8]**

---

## 🔍 SLIDE 8: ABLATION STUDIES
**Duration: 25 seconds | Cumulative: 4:00**
**Speaker: NAVIN**

---

**[NAVIN - Analytical tone]**

"Our ablation studies revealed three key insights.

First, for the Sobolev regularization weight lambda, we swept values from zero point zero zero one to zero point one. Lambda equals zero point one achieved the lowest training loss, suggesting stronger smoothness constraints aid convergence.

Second, comparing adaptive versus uniform sampling, the adaptive variant reduced validation error by twelve percent and improved Gamma smoothness by five percent. You can see the smoother loss curves here.

Third, and most important, the physics-informed loss was critical. Without the PDE term, out-of-distribution performance dropped by eighteen percent. This confirms that embedding domain knowledge beats pure data-driven learning for extrapolation beyond the training distribution. Andrew will discuss our limitations."

**[Transition to Slide 9]**

---

## ⚠️ SLIDE 9: LIMITATIONS & FUTURE WORK
**Duration: 25 seconds | Cumulative: 4:25**
**Speaker: ANDREW**

---

**[ANDREW - Honest, constructive tone]**

"Our main limitation is higher-order Greeks. Theta, Vega, and Rho still have errors eight to forty times above our target thresholds. This tells us the current loss weighting under-emphasizes these derivatives.

We also see two to three times higher errors near expiration, where the PDE becomes mathematically stiff and the payoff has a discontinuous derivative at the strike price.

For future work, we should explore explicit Greek supervision terms—adding direct loss terms for Theta, Vega, and Rho. We could also try multi-task learning with separate network heads for each Greek.

Beyond this, extending the framework to American options with early exercise, stochastic volatility models like Heston, and multi-asset baskets would test whether PINNs can scale to production derivatives workloads.

Finally, hybrid approaches combining PINNs for speed with traditional finite differences for critical calculations requiring machine precision could bridge the gap to real trading systems. Rahul will conclude."

**[Transition to Slide 10]**

---

## 🎯 SLIDE 10: CONCLUSIONS & IMPACT
**Duration: 25 seconds | Cumulative: 4:50**
**Speaker: RAHUL**

---

**[RAHUL - Strong, confident conclusion]**

"To conclude: We demonstrated that a single physics-informed neural network can generalize across volatility regimes without retraining, achieving excellent first-order Greek accuracy. Our Delta and Gamma errors are well below target thresholds and competitive with Monte Carlo.

Our key contributions are: First, single-model generalization across the full volatility range with less than ten percent out-of-distribution degradation. Second, eighteen percent improved extrapolation from physics-informed regularization. Third, seventeen percent training speedup from adaptive sampling. And fourth, smooth Gamma surfaces suitable for real hedging.

The impact? Market makers and risk managers could evaluate thousands of scenarios in real-time instead of waiting hours for Monte Carlo simulations. While challenges remain for higher-order Greeks, our results show PINNs are a promising direction for modern quantitative finance, balancing speed, accuracy, and theoretical consistency.

Thank you for your attention. We're happy to answer any questions!"

**[END OF PRESENTATION - 4:50]**

---

## 🎬 PRODUCTION NOTES

### Pacing Guide
- **Andrew:** Moderate pace, welcoming tone (3 sections: intro, setup, limitations)
- **Rahul:** Slightly faster, enthusiastic (3 sections: problem, results, conclusion)
- **Aditya:** Technical but clear, point to visuals (2 sections: architecture, visualizations)
- **Navin:** Analytical, emphasize insights (2 sections: physics loss, ablations)

### Emphasis Points

**Key numbers to stress:**
- "zero point zero zero eight five" (Delta MAE)
- "zero point zero zero zero five three" (Gamma MAE)
- "**eighteen** percent improvement" (OOD generalization)
- "**seventeen** percent faster" (adaptive sampling)

**Words to emphasize:**
- "physics-informed" (not just data-driven)
- "single model" (no retraining)
- "real-time" (practical impact)
- "smooth" (for hedging quality)

### Camera/Visual Notes

**Slide 2:** Pause briefly after "research question" to let it sink in

**Slide 4:** Point to PDE equation and heatmap when mentioning them

**Slide 6:** Emphasize the checkmarks (✓) for what worked, acknowledge limitations honestly

**Slide 7:** Gesture to each panel (left/center/right) as you describe them

**Slide 10:** Make eye contact with camera during conclusion

### Transition Phrases (Critical for Smooth Flow)

After each section, the current speaker explicitly hands off:

1. Andrew → Rahul: "Rahul will start by explaining the problem"
2. Rahul → Aditya: "Aditya will now explain our technical approach"
3. Aditya → Navin: "Navin will explain what makes this physics-informed"
4. Navin → Andrew: "Andrew will now describe our experimental setup"
5. Andrew → Rahul: "Rahul will now present our results"
6. Rahul → Aditya: "Aditya will show what the model learned visually"
7. Aditya → Navin: "Navin will walk through our ablation studies"
8. Navin → Andrew: "Andrew will discuss our limitations"
9. Andrew → Rahul: "Rahul will conclude"

### Backup Time-Saving Cuts (If Running Over)

**If at 3:00 and behind schedule, cut:**
- Slide 4 (Navin): Reduce to 20 sec, skip heatmap description (-10 sec)
- Slide 9 (Andrew): Mention only 2 limitations instead of 3 (-10 sec)

**If at 4:00 and behind schedule, cut:**
- Slide 8 (Navin): Skip lambda sweep details, just show adaptive sampling (-10 sec)
- Slide 6 (Rahul): Skip Theta/Vega details, just say "higher-order Greeks need work" (-10 sec)

**Never cut:**
- Main results table (Slide 6)
- Hypothesis validation
- Key visualizations (Slide 7)
- Impact statement (Slide 10)

---

## 📊 WORD COUNT & SPEAKING RATE

| Speaker | Words | Duration | Rate (wpm) | Slides |
|---------|-------|----------|------------|--------|
| Andrew  | ~170  | 1:05     | ~156       | 1, 5, 9 |
| Rahul   | ~285  | 1:45     | ~163       | 2, 6, 10 |
| Aditya  | ~195  | 1:10     | ~167       | 3, 7 |
| Navin   | ~160  | 0:55     | ~174       | 4, 8 |
| **Total** | **~810** | **4:55** | **~164** | **10** |

**Target speaking rate:** 150-170 words per minute (conversational pace)

---

## ✅ FINAL CHECKLIST

### Before Recording
- [ ] Each person has printed their sections
- [ ] Practiced at least 2 times with timer
- [ ] Slides finalized with all figures embedded
- [ ] Quiet recording location confirmed
- [ ] Audio equipment tested (headset/mic)

### During Recording
- [ ] Timer visible to all speakers
- [ ] Smooth transitions between speakers
- [ ] Clear enunciation, moderate pace
- [ ] Point to visuals when referencing them
- [ ] Smile at camera (sounds better!)

### After Recording
- [ ] Video is under 5:00 ✓
- [ ] Audio is clear for all speakers ✓
- [ ] All figures are visible ✓
- [ ] All 4 team members participated ✓
- [ ] No "um", "uh", long pauses ✓

### Before Submission
- [ ] Watched full video together
- [ ] Checked playback on Gradescope
- [ ] Verified all team members added to group
- [ ] Submitted at least 1 hour before deadline

---

## 🎓 FINAL CONFIDENCE BOOST

**You have:**
✅ Excellent technical results (Delta/Gamma both exceed targets)
✅ All 3 hypotheses validated
✅ Honest discussion of limitations
✅ Clear practical impact
✅ Strong visualizations

**Just deliver it clearly in <5 minutes and you'll score 95+/100!**

**Break a leg! 🎬**
