---
marp: true
title: Tensor Titans - WiDS Worldwide Global Datathon 2026
paginate: true
theme: default
style: |
  section {
    font-family: Arial, Helvetica, sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f2937 100%);
    color: #f8fafc;
  }
  h1, h2, h3 {
    color: #fbbf24;
  }
  strong {
    color: #93c5fd;
  }
  ul {
    margin-top: 0.4em;
  }
  section.title {
    text-align: center;
  }
  section.title h1 {
    font-size: 2.1em;
    margin-bottom: 0.2em;
  }
  section.title h2 {
    font-size: 1.1em;
    color: #e5e7eb;
  }
---

<!-- _class: title -->
# Tensor Titans
## WiDS Worldwide Global Datathon 2026
## Wildfire Survival Analysis

**Multi-Horizon Threat Forecasting**

---

# 1. Wildfire Survival Analysis Challenge

## What We Delivered

- Understood **right-censored survival analysis** framework
- Analyzed early-incident signals from first 5 hours after fire ignition
- Modeled threat to evacuation zones within 72-hour window
- Designed multi-horizon probability forecasts (12h, 24h, 48h, 72h)

## Key Challenge

- **Event:** Fire within 5 km of evacuation zone centroid
- **Censoring:** ~80-85% of fires do not threaten evacuation zones
- **Goal:** Rank fires by urgency AND generate calibrated probabilities

---

# 2. Survival Models and Evaluation Metric

## What We Delivered

- Explored wildfire perimeter dynamics and spatial relationships
- Built two baseline survival models:
  - Logistic regression (probabilistic predictions)
  - Feedforward neural network (multi-horizon outputs)
- Standardized features using `StandardScaler`
- Trained with stratified train-validation split

## Hybrid Evaluation Metric

- **C-Index (30%):** Ranking quality - how well fires ordered by urgency
- **Weighted Brier Score (70%):** Probability calibration
  - 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h
- 48h weighted highest (strongest operational value)

---

# 3. Submission and Result

## What We Delivered

- Generated multi-horizon threat probabilities (12h, 24h, 48h, 72h)
- Formatted submission with four probability columns per fire
- Submitted to Kaggle leaderboard

## Final Result

- **Hybrid Score: 0.82691**
  - Strong ranking performance (C-index component)
  - Well-calibrated probabilities (Brier Score component)
- Baseline pipeline effective for operational emergency response
- Future: Advanced survival models, probability calibration, ensemble methods

## Closing Line

That is our deliverable summary for wildfire survival analysis.