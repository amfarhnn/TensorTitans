# � WiDS Global Datathon 2026 - Wildfire Survival Analysis Challenge

⚡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚡
                 T E N S O R    T I T A N S
⚡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚡

## 📋 Overview

**TensorTitans** is a competitive machine learning project developed for the **WiDS Global Datathon 2026**. This project tackles a **survival analysis challenge** to predict wildfire threat probability across multiple time horizons. Using early wildfire signals from the first 5 hours after ignition, we build calibrated probability forecasts to support emergency response decisions. Our team achieved a **Hybrid Score of 0.82691** on the competition leaderboard.

## 👥 Team Members

| Member | ID |
|--------|-----|
| Amir Farhan bin Ghaffar | 2115617 |
| Muhammad Irsyad Ilham bin Azizan | 2217555 |
| Muhammad Amin bin Mohamad Rizal | 2217535 |

**Project Track:** Track 1 - Kaggle Competition

**Competition Link:** [WiDS Worldwide Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

**Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vBPurjaW-FoK5wEFkfzQJPGDaxYMG1UM)

---

## 🎯 Project Objectives

The WiDS Global Datathon 2026 challenges participants to:
- Predict wildfire threat probability across multiple time horizons (12h, 24h, 48h, 72h)
- Apply survival analysis techniques to right-censored fire data
- Generate calibrated probability forecasts for emergency response decisions
- Rank fires by urgency to support triage and resource allocation
- Optimize for both ranking performance (C-index) and probability calibration (Brier Score)

---

## 📊 Dataset Overview

### Files Included

| File | Description |
|------|-------------|
| `train.csv` | Training dataset with wildfire features and survival targets |
| `test.csv` | Test dataset with features only (for predictions) |
| `metaData.csv` | Feature descriptions and variable meanings |
| `sample_submission.csv` | Required submission format with multi-horizon probabilities |

### Dataset Characteristics

- **Type:** Right-censored survival analysis data
- **Target Variable:** Multi-horizon threat prediction (12h, 24h, 48h, 72h)
- **Event Definition:** Fire within 5 km of evacuation zone centroid
- **Feature Window:** First 5 hours after initial perimeter observation
- **Censoring:** Fires not reaching evacuation zone within 72h are right-censored
- **Feature Types:** Wildfire perimeter dynamics and spatial relationships

---

## 📈 Evaluation Metric: Hybrid Score

**Hybrid Score = 0.3 × C-index + 0.7 × (1 - Weighted Brier Score)**

### C-Index (30%)
- Measures ranking quality: how well fires are ordered by urgency
- Range: 0.5 to 1.0 (higher is better)
- Critical for triage and resource prioritization

### Weighted Brier Score (70%)
- Measures probability calibration at multiple horizons
- Uses censor-aware evaluation:
  - **Hits:** 1 if threatened by horizon H, else 0
  - **Censored after H:** 0
  - **Censored before H:** excluded
- Weighted average: 0.3 × Brier@24h + 0.4 × Brier@48h + 0.3 × Brier@72h
- 48h weighted highest because it balances lead time with operational urgency

---

## 🏆 Competition Results

### Leaderboard Performance

- **Hybrid Score:** 0.82691
  - C-index component: 30% of score
  - Brier Score component: 70% of score
- **Rank:** #1 (First Entry)
- **Status:** Successfully submitted

![Leaderboard Screenshot](Leaderboard.png)

---

## 🧠 Model Architecture & Approach

### Survival Analysis Framework

The models generate calibrated survival probabilities across multiple time horizons:
- **12-hour forecast:** For immediate response decisions
- **24-hour forecast:** For near-term alerts and preparation
- **48-hour forecast:** Primary operational window (weighted 40% in evaluation)
- **72-hour forecast:** Extended planning horizon

### Baseline Models

1. **Logistic Regression Pipeline**
   - Standardized features with `StandardScaler`
   - Logistic regression for probability estimates
   - Calibration via probability scaling

2. **Feedforward Neural Network (MLP)**
   - Multiple dense layers with ReLU activation
   - Output layer with sigmoid activation per time horizon
   - Handles nonlinear relationships in early-incident signals

### Model Objectives

- **Ranking:** Correctly order fires by threat urgency (C-index)
- **Calibration:** Produce trustworthy probability estimates (Brier Score)
- **Multi-horizon:** Forecast across 12h, 24h, 48h, and 72h windows

---

## 📝 Key Implementation Steps

### 1. Data Inspection and Censoring
- Load training/test datasets with survival times and event indicators
- Understand censoring structure (right-censored at 72 hours)
- Identify fires that threatened evacuation zones within the observation window

### 2. Exploratory Data Analysis (EDA)
- Analyze early-incident signal distributions (first 5 hours)
- Examine fire perimeter dynamics and spatial relationships
- Study threat event timing and censoring patterns
- Compute correlations between early signals and threat probability

### 3. Feature Engineering
- Engineer features from wildfire perimeter data
- Standardize numerical features with `StandardScaler`
- Handle missing values appropriately
- Create lagged and rate-of-change features from early signals

### 4. Survival Model Training
- Train-validation split with stratification by event status
- Implement baseline and deep learning survival models
- Generate multi-horizon probability predictions (12h, 24h, 48h, 72h)
- Tune for both ranking (C-index) and calibration (Brier Score)

### 5. Evaluation & Multi-Horizon Submission
- Calculate C-index for ranking quality
- Calculate censor-aware Brier Scores for each horizon
- Compute hybrid score: 0.3 × C-index + 0.7 × (1 - Weighted Brier)
- Format submission with probability columns for each time horizon

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.8+
PyTorch
Pandas
Scikit-learn
Numpy
Matplotlib
Seaborn
```

### Installation
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn
```

### Usage

1. **Exploratory Analysis:**
   - Open `TensorTitans.ipynb`
   - Run the EDA section to understand dataset characteristics

2. **Model Training:**
   - Execute the baseline model training section
   - Monitor validation AUC during training

3. **Prediction & Submission:**
   - Generate predictions on test set
   - Format according to `sample_submission.csv`
   - Submit to Kaggle competition

---

## 📁 Project Structure

```
TensorTitans/
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── metaData.csv             # Feature metadata
├── sample_submission.csv    # Submission template
├── TensorTitans.ipynb       # Main notebook
├── TensorTitans (1).ipynb   # Alternative/backup notebook
├── Leaderboard.png          # Competition leaderboard screenshot
├── ReadMe.md                # This file
└── hi.txt                   # Additional notes
```

---

## 🔍 Key Findings

### Data Insights
- Wildfire threat events are relatively rare (~15-20% threaten evacuation zones)
- Early signals (first 5 hours) have predictive power for 72-hour threat window
- Perimeter growth rate and spatial proximity are key indicators
- Feature distributions vary widely, requiring standardization
- Right-censoring structure must be carefully handled in evaluation

### Model Performance
- Baseline logistic regression provides solid ranking and calibration baseline
- Deep learning survival models improve calibration and handling of nonlinearities
- **Hybrid Score:** 0.82691 (C-index + Weighted Brier Score)
- Multi-horizon predictions align with operational decision windows
- 48h forecast is most valuable for emergency response

---

## 💡 Future Improvements

1. **Advanced Survival Models:** Cox proportional hazards, Random survival forests, Gradient boosting survival
2. **Probability Calibration:** Apply isotonic regression or Platt scaling to improve Brier Score
3. **Temporal Feature Engineering:** Extract more sophisticated signals from fire dynamics
4. **Ensemble Methods:** Combine survival models for improved ranking and calibration
5. **Hyperparameter Tuning:** Optimize for both C-index and Brier Score components
6. **Cross-Validation:** Stratified k-fold with proper event/censor handling
7. **Real-World Validation:** Test with held-out fire seasons for operational readiness

---

## 📚 References

- [WiDS Datathon 2026 Competition](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn Guide](https://scikit-learn.org/)

---

## 📝 License & Attribution

This project is developed for educational purposes as part of the WiDS Global Datathon 2026 competition.

---

**Last Updated:** May 5, 2026  
**Status:** Completed & Submitted ✅
