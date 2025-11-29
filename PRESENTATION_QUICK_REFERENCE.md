# Video Presentation Quick Reference Card

## ⏱️ TIMING CHEAT SHEET (Print this!)

| Slide | Speaker | Duration | End Time | Topic |
|-------|---------|----------|----------|-------|
| 1 | Andrew | 0:15 | 0:15 | Title & Intro |
| 2 | Rahul | 0:35 | 0:50 | Problem & Motivation |
| 3 | Aditya | 0:45 | 1:35 | Architecture |
| 4 | Navin | 0:30 | 2:05 | Physics-Informed Loss |
| 5 | Andrew | 0:20 | 2:25 | Experimental Setup |
| 6 | Rahul | 0:45 | 3:10 | Main Results |
| 7 | Aditya | 0:25 | 3:35 | Visualizations |
| 8 | Navin | 0:25 | 4:00 | Ablations |
| 9 | Andrew | 0:25 | 4:25 | Limitations |
| 10 | Rahul | 0:25 | 4:50 | Conclusion |

**⚠️ CRITICAL: Stay under 5:00 or lose 20-50% of grade!**

---

## 🎤 SPEAKER ASSIGNMENTS

### Andrew (1:05 total)
- Slide 1: Welcome & team intro
- Slide 5: Dataset and hypotheses
- Slide 9: Limitations & future work

**Key phrase:** "Our approach addresses this by treating volatility as an input..."

---

### Rahul (1:45 total)
- Slide 2: Problem motivation & research question
- Slide 6: Main results & hypothesis validation
- Slide 10: Conclusions & impact

**Key phrase:** "Our results confirm all three hypotheses..."

---

### Aditya (1:10 total)
- Slide 3: Network architecture
- Slide 7: Visualizations

**Key phrase:** "The magic happens in our loss function..."

---

### Navin (0:55 total)
- Slide 4: Physics-informed loss & PDE
- Slide 8: Ablation studies

**Key phrase:** "The PDE residual term forces the network to satisfy..."

---

## 📊 FIGURES TO INCLUDE

**MUST HAVE:**
1. `figures/final_results/pinn_surface_3d.png` (Slide 7, left)
2. `figures/end_to_end/oos/pinn_delta_surface.png` (Slide 7, center-top)
3. `figures/end_to_end/oos/pinn_gamma_surface.png` (Slide 7, center-bottom)
4. `figures/residual_heatmaps/pde_residual_coarse.png` (Slide 4 & 7, right)

**NICE TO HAVE:**
5. Training curves (Slide 8)
6. Comparison bar charts (Slides 6, 8)

---

## 🎯 KEY NUMBERS TO MEMORIZE

**Main Results:**
- Delta MAE: **0.0085** (target <0.05 ✓)
- Gamma MAE: **0.00053** (target <0.10 ✓)
- Gamma TV ratio: **0.98** (target <2.0 ✓)

**Hypothesis Validation:**
- H1: ✓ Meets targets across σ ∈ [0.05, 0.60]
- H2: **18%** OOD improvement vs supervised
- H3: **17%** faster convergence with adaptive sampling

**Dataset:**
- **1M** training, 100K val, 100K test
- **100** epochs
- σ ∈ [0.05, 0.60] train, [0.60, 0.65] OOD test

---

## 💡 TRANSITIONS BETWEEN SPEAKERS

**Rahul → Aditya (after Slide 2):**
> "Now Aditya will explain our technical approach."

**Aditya → Navin (after Slide 3):**
> "Navin will now explain what makes this physics-informed."

**Navin → Andrew (after Slide 4):**
> "Andrew will describe our experimental setup."

**Andrew → Rahul (after Slide 5):**
> "Rahul will present our main results."

**Rahul → Aditya (after Slide 6):**
> "Aditya will show our learned surfaces."

**Aditya → Navin (after Slide 7):**
> "Navin will walk through our ablation studies."

**Navin → Andrew (after Slide 8):**
> "Andrew will discuss limitations and future work."

**Andrew → Rahul (after Slide 9):**
> "And finally, Rahul will conclude."

---

## ✅ PRE-RECORDING CHECKLIST

### 3 Days Before
- [ ] Slides finalized
- [ ] Script distributed to all team members
- [ ] Practice run #1 (expect to go over time)
- [ ] Identify sections to cut

### 1 Day Before
- [ ] Practice run #2 (should be under 5:00)
- [ ] Test recording setup (audio, video, screen share)
- [ ] Schedule recording session

### Day of Recording
- [ ] Quiet location secured
- [ ] Good lighting (face windows or lamps)
- [ ] Headset/microphone tested
- [ ] Zoom/recording software ready
- [ ] All team members present 15 min early

### After Recording
- [ ] Watch full video
- [ ] Check audio quality (all speakers clear?)
- [ ] Check timing (under 5:00?)
- [ ] Check figures (all visible and readable?)
- [ ] Re-record if any major issues

### Before Submission
- [ ] Upload to Gradescope
- [ ] Add all team members to group
- [ ] Verify video plays on Gradescope
- [ ] Watch uploaded version one final time

---

## 🚨 EMERGENCY TIME-SAVING CUTS

**If running over time, cut these in order:**

1. **Slide 9:** Reduce future work from 8 items to 4 (-10 sec)
2. **Slide 6:** Skip Vega/Rho details, focus on Delta/Gamma (-10 sec)
3. **Slide 4:** Simplify PDE explanation (-10 sec)
4. **Slide 8:** Show only 2 ablations instead of 3 (-10 sec)
5. **Slide 2:** Shorten related work discussion (-10 sec)

**DO NOT CUT:**
- Main results (Slide 6)
- Hypothesis validation
- Key visualizations (Slide 7)
- Limitations (Slide 9)

---

## 🎬 RECORDING SETUP OPTIONS

### Option A: Zoom (Easiest)
1. Schedule Zoom meeting
2. All join, share screen with slides
3. Record to local computer
4. Export as MP4

**Pros:** Natural, shows teamwork
**Cons:** Harder to edit mistakes

---

### Option B: Individual Recording + Editing
1. Each person records their sections separately
2. Use OBS Studio or PowerPoint recording
3. Combine in video editor (DaVinci Resolve, iMovie)

**Pros:** Can re-record mistakes easily
**Cons:** More technical, less natural flow

---

### Option C: Narrated Slides (Simplest)
1. One person shares screen
2. Others join audio-only
3. Record via Zoom or PowerPoint

**Pros:** Clean, simple
**Cons:** Less engaging

---

## 📱 TECH REQUIREMENTS

**Video:**
- Resolution: 1920x1080 minimum
- Format: MP4 (H.264 codec)
- Frame rate: 30 fps minimum

**Audio:**
- Sample rate: 44.1 kHz
- Bit depth: 16-bit minimum
- **NO** background noise, echo, or clipping

**File Size:**
- Gradescope limit: Check before submission
- Typically <500 MB for 5 min video

---

## 🏆 GRADING OPTIMIZATION

### Logistics (10 pts)
✓ All 4 team members speak ~1 min each
✓ Smooth transitions between speakers

### Presentation Quality (20 pts)
**Visuals (10 pts):**
✓ High contrast (dark text, light background)
✓ Large fonts (≥24pt body, ≥32pt titles)
✓ Clear figures from report
✓ Simple layouts (3-5 bullets max)

**Speaking (10 pts):**
✓ Clear enunciation, moderate pace
✓ No "um", "uh", excessive pauses
✓ Enthusiastic tone
✓ Good audio quality

### Content (70 pts)
**Problem (15 pts):**
✓ Clear Greeks computation challenge
✓ Real-world importance stated
✓ Gap in prior work identified

**Approach (20 pts):**
✓ Architecture explained
✓ Physics-informed loss detailed
✓ Key innovation (σ as input) highlighted
✓ Adaptive sampling described

**Results (20 pts):**
✓ Main table with all metrics
✓ All 3 hypotheses validated
✓ Ablations shown
✓ Limitations discussed honestly

**Conclusions (15 pts):**
✓ 4 contributions summarized
✓ Impact statement clear
✓ Future work concrete

---

## 💬 SAMPLE OPENING (Andrew)

> "Hi, I'm Andrew. Today our team presents 'Greeks Estimation via Physics-Informed Neural Networks.' We tackled a critical challenge in quantitative finance: computing option price sensitivities—called Greeks—which are essential for risk management but computationally expensive. I'm joined by Rahul, Aditya, and Navin. Rahul will explain the problem."

**Timing: 15 seconds**

---

## 💬 SAMPLE CLOSING (Rahul)

> "To conclude: We demonstrated that a single physics-informed neural network can generalize across volatility regimes without retraining, achieving excellent first-order Greek accuracy. The PDE constraint improves extrapolation by 18%, and adaptive sampling accelerates training by 17%. Market makers and risk managers could evaluate thousands of scenarios in real-time. While challenges remain for higher-order Greeks, our results show PINNs are a promising direction for modern quantitative finance. Thank you!"

**Timing: 25 seconds**

---

## 🔥 LAST-MINUTE TIPS

1. **Smile at the camera** - You sound better when smiling!
2. **Slow down 10%** - You naturally speed up when nervous
3. **Pause between sections** - Easier to edit later
4. **Have water nearby** - Clear your throat before starting
5. **Stand if possible** - Better energy and voice projection
6. **Look at camera, not screen** - More engaging for viewers
7. **Practice transitions** - Avoid awkward pauses between speakers

---

## 📞 Emergency Contact

If technical issues on submission day:
1. Try different browser for Gradescope upload
2. Check file format (MP4 is safest)
3. Compress video if too large (HandBrake software)
4. Post on Piazza if persistent issues

---

## 🎓 FINAL CONFIDENCE BOOST

**Your project is excellent!**

- Delta MAE 0.0085 beats target by 6×
- Gamma MAE 0.00053 beats target by 200×
- All 3 hypotheses validated
- Honest about limitations
- Clear future work

**You have the content. Now just deliver it clearly in <5 minutes!**

Good luck! 🚀
