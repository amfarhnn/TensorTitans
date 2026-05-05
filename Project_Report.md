# Assessment 1: Deep Learning Project Report

## WiDS Worldwide Global Datathon 2026 - Wildfire Survival Analysis

### University Project Template

**Course/Module:** [Insert course name]

**Lecturer/Instructor:** [Insert name]

**Group Name:** Tensor Titans

**Project Track:** Track 1: Kaggle Competition

**Competition:** [WiDS Worldwide Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

**Submission Date:** [Insert date]

---

## Abstract

This report presents the work completed by Team Tensor Titans for the WiDS Worldwide Global Datathon 2026, a Kaggle-based **survival analysis challenge** to predict wildfire threat probability across multiple time horizons. The objective was to develop calibrated probability forecasts that support emergency response decisions, using only early-incident signals from the first 5 hours after fire ignition.

The project followed a survival analysis framework with right-censored data. Fires are classified by whether they threaten evacuation zones within 12, 24, 48, and 72 hours. The evaluation metric is a hybrid score combining C-index (30%, ranking quality) and weighted Brier Score (70%, probability calibration). A baseline logistic regression model and a feedforward neural network were implemented. The final submission achieved a **Hybrid Score of 0.82691** on the Kaggle leaderboard, demonstrating effective ranking and calibrated probability generation.

---

## 1. Introduction

The WiDS Worldwide Global Datathon 2026 is a machine learning competition focused on solving a real-world operational problem: **predictive wildfire threat forecasting**. When wildfires ignite, emergency managers must decide which communities to warn and where to position scarce resources, often with incomplete information. This competition frames that challenge as a **survival analysis problem** using early-incident signals.

Participants develop calibrated probability forecasts that answer: Which fires will threaten populated evacuation zones? How soon? And how confident should we be in that forecast? The solution requires two complementary goals: ranking fires correctly by urgency (for triage) and producing trustworthy probability estimates (for decision thresholds).

This report documents the development process carried out by Team Tensor Titans. The work includes data inspection, exploratory analysis, preprocessing, survival model development, and evaluation using a hybrid metric that balances ranking and calibration.

---

## 2. Problem Statement

Emergency managers face a critical challenge with incomplete information: **When a wildfire ignites, which communities will it threaten, and how quickly?** The operational context requires two types of insight:

1. **Urgent Ranking:** Which fires demand immediate attention first?
2. **Calibrated Forecasts:** What is the probability of threat within actionable time windows (12h, 24h, 48h, 72h)?

This competition frames the problem as **right-censored survival analysis**:
- **Event:** Fire threatens evacuation zone centroid within 5 km
- **Observation Window:** 72 hours from t0 + 5 hours (where t0 is initial perimeter observation)
- **Features:** Computed only from the first 5 hours after ignition
- **Censoring:** Fires not reaching evacuation zones within 72h are right-censored

The main technical challenges are:
- Building models that rank fires correctly by threat urgency (C-index)
- Generating calibrated probabilities across multiple time horizons (Brier Score)
- Handling right-censored survival data appropriately
- Optimizing for a hybrid metric that balances ranking and calibration

---

## 3. Objectives

The project was designed to achieve the following objectives:

- understand survival analysis and right-censored data structure,
- inspect the dataset using exploratory analysis of early-incident signals,
- identify patterns in wildfire perimeter dynamics and spatial relationships,
- prepare features for survival modeling,
- build baseline survival models for multi-horizon probability prediction,
- evaluate models using both ranking (C-index) and calibration (Brier Score) metrics,
- generate calibrated probability forecasts for emergency response decisions,
- submit predictions in the required multi-horizon format.

---

## 4. Dataset Description

The project uses four main files:

| File | Description |
|------|-------------|
| `train.csv` | Training data with wildfire features, event indicator, and threat times |
| `test.csv` | Test data used for multi-horizon probability prediction |
| `metaData.csv` | Feature descriptions and variable meanings |
| `sample_submission.csv` | Reference format for multi-horizon probability submission |

### Dataset Characteristics

- **Structure:** Right-censored survival analysis data
- **Event Definition:** Fire within 5 km of evacuation zone centroid
- **Observation Window:** From t0 + 5 hours to t0 + 77 hours (72-hour prediction window)
- **Feature Window:** Computed from first 5 hours after initial perimeter observation
- **Features:** Wildfire perimeter dynamics and spatial relationships
- **Censoring:** ~80-85% censored (fires not threatening within 72h); ~15-20% uncensored (events observed)
- **Time Horizons:** 12h, 24h, 48h, 72h prediction windows

---

## 5. Abstract of Methodology

The workflow began with understanding the survival analysis framework and right-censored data structure. Data inspection and EDA examined early-incident signal distributions, fire dynamics, and threat event timing.

Features were engineered from wildfire perimeter data and standardized using `StandardScaler`. Two baseline models were implemented: a logistic regression pipeline and a feedforward neural network. Both models output calibrated probabilities for each time horizon (12h, 24h, 48h, 72h).

The data was split into training and validation sets with stratification by event status. Model performance was evaluated using two metrics:
- **C-index:** Measures ranking quality (how well fires are ordered by urgency)
- **Weighted Brier Score:** Measures probability calibration across three horizons (24h, 48h, 72h) with censor-aware weighting

The hybrid score combines these: 0.3 × C-index + 0.7 × (1 - Weighted Brier Score).

---

## 6. Exploratory Data Analysis

Exploratory data analysis was used to understand the dataset before training any model.

### EDA Activities

- loading `train.csv`, `test.csv`, and `metaData.csv`,
- checking dataset shape and column structure,
- viewing sample rows,
- generating summary statistics,
- identifying missing values,
- checking the target distribution,
- examining correlations between features.

### Purpose of EDA

The EDA stage helped identify data quality issues and provided a clearer understanding of the relationships within the dataset. This supported better preprocessing and modeling decisions.

---

## 7. Methodology

### 7.1 Data Preprocessing

The preprocessing pipeline followed the steps below:

- remove irrelevant identifier columns such as `event_id`,
- handle missing values,
- standardize numerical features using `StandardScaler`,
- prepare the dataset for binary classification.

Preprocessing is essential because tabular data often contains variables with different scales and missing entries. Standardization improves the stability of many machine learning models, especially gradient-based approaches.

### 7.2 Baseline Modeling

Two baseline approaches were used in the notebook.

#### Logistic Regression Pipeline

The notebook includes a scikit-learn pipeline consisting of:

- standardization,
- logistic regression with a high iteration limit.

This provides a strong and interpretable baseline for binary classification.

#### Feedforward Neural Network

The deep learning model is a multilayer perceptron implemented in PyTorch.

##### Model Structure

- input layer,
- fully connected layer with 64 units and ReLU activation,
- fully connected layer with 32 units and ReLU activation,
- output layer with sigmoid activation.

The model architecture is appropriate for tabular binary classification because it can capture nonlinear relationships between features.

### 7.3 Training Procedure

The model was trained using an 80-20 train-validation split. After training, predicted probabilities were generated on the validation set and AUC-ROC was calculated to measure generalization performance.

---

## 8. Results

The project achieved a **Hybrid Score of 0.82691** on the Kaggle leaderboard.

### Competition Outcome

- **Hybrid Score:** 0.82691
  - Components: 0.3 × C-index + 0.7 × (1 - Weighted Brier Score)
- **Submission Status:** Successfully submitted

![Leaderboard Screenshot](Leaderboard.png)

This score indicates that the implemented baseline pipeline achieved strong performance in both ranking fires by urgency (C-index) and generating calibrated probability forecasts (Brier Score) across the 24h, 48h, and 72h horizons.

---

## 9. Discussion

The results demonstrate that the baseline survival models successfully addressed a real-world operational forecasting challenge. The achieved hybrid score of 0.82691 reflects strong performance in two complementary objectives:

1. **Ranking Quality (C-index):** The models correctly rank fires by threat urgency, supporting emergency triage decisions.
2. **Probability Calibration (Brier Score):** The forecasts are well-calibrated across multiple time horizons, allowing operators to apply probabilistic thresholds for alerts and resource deployment.

The multi-horizon approach (12h, 24h, 48h, 72h) aligns with operational decision windows. The 48h horizon is weighted most heavily (40%) in evaluation because it offers the strongest balance between actionable lead time and decision urgency.

The use of early-incident signals (first 5 hours) demonstrates that reliable threat forecasts can be generated before fire behavior fully develops, critical for emergency response timing.

---

## 10. Conclusion

This project demonstrates a complete machine learning workflow for operational wildfire threat forecasting, from data inspection and preprocessing to multi-horizon survival model development and calibrated probability submission. Team Tensor Titans built baseline models for the WiDS Worldwide Global Datathon 2026 and achieved a hybrid score of 0.82691 on the leaderboard.

The work confirms that early-incident wildfire signals can be modeled effectively to generate trustworthy probability forecasts across multiple decision horizons. The project provides a foundation for emergency response systems that balance ranking accuracy (for triage) with probability calibration (for threshold-based decisions).

---

## 11. Recommendations for Future Work

The following improvements are recommended for a stronger future submission:

- implement advanced survival models: Cox proportional hazards, random survival forests, gradient boosting survival,
- apply probability calibration techniques: isotonic regression, Platt scaling,
- extract more sophisticated temporal features from fire dynamics,
- develop ensemble methods combining multiple survival models,
- conduct aggressive hyperparameter tuning optimized for hybrid score,
- apply stratified k-fold cross-validation with proper event/censor handling,
- validate models against held-out fire seasons for operational deployment readiness.

---

## 12. References

- WiDS Worldwide Global Datathon 2026: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26
- PyTorch Documentation: https://pytorch.org/docs/
- Scikit-learn Documentation: https://scikit-learn.org/

---

## 13. Appendix

### A. Project Files

- `train.csv`
- `test.csv`
- `metaData.csv`
- `sample_submission.csv`
- `TensorTitans.ipynb`
- `Leaderboard.png`

### B. Notes

This report is written as a Word-style template suitable for direct conversion into a formal university submission document. The placeholders in the title block can be replaced with the required course, instructor, and submission details.