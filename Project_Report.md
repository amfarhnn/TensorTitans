# Assessment 1: Deep Learning Project Report

## WiDS Worldwide Global Datathon 2026 - Wildfire Survival Analysis

**Group Name:** Tensor Titans  
**Project Track:** Track 1 - Kaggle Competition  
**Competition:** WiDS Worldwide Global Datathon 2026  
**Submission Date:** June 17, 2026  

### Team Members

| Member | ID |
|---|---|
| Amir Farhan bin Ghaffar | 2115617 |
| Muhammad Irsyad Ilham bin Azizan | 2217555 |
| Muhammad Amin bin Mohamad Rizal | 2217535 |

---

## Abstract

This report presents Team Tensor Titans' machine learning project for the WiDS Worldwide Global Datathon 2026. The competition focuses on wildfire threat forecasting using survival analysis. The goal is to predict the probability that a wildfire will come within 5 km of an evacuation-zone centroid within four operational time horizons: 12 hours, 24 hours, 48 hours, and 72 hours.

The dataset contains engineered wildfire features computed from the first five hours after initial fire observation. These features describe fire growth, movement, distance to evacuation zones, directional alignment, and event start time. The target is right-censored because many fires do not reach an evacuation zone during the 72-hour observation window.

The project includes data inspection, feature understanding, exploratory analysis, censor-aware target construction, multi-horizon model training, validation using ranking and calibration metrics, and Kaggle submission formatting. The improved notebook, `TensorTitans_Improved.ipynb`, implements a reproducible multi-horizon baseline using regularized logistic regression blended with histogram gradient boosting. The final project documentation also includes a plain-language dataset explanation, README, presentation materials, and this full report.

The team recorded a Kaggle hybrid score of 0.82691. The improved notebook strengthens the original workflow by using separate censor-aware labels for each prediction horizon instead of assigning one probability to all horizons.

---

## 1. Introduction

Wildfires are dangerous because their early movement is uncertain and emergency managers must make decisions before complete information is available. When a fire begins, authorities need to know which communities may be threatened, how soon that threat may occur, and how confident the warning should be. This competition frames that problem as a machine learning survival-analysis task.

In this project, each row represents one wildfire event. The model receives only information from the first five hours after the initial fire perimeter observation. From those early signals, it must estimate the probability that the fire will reach the threat definition within 12, 24, 48, or 72 hours. A fire is considered a threat when it comes within 5 km of the center of an evacuation zone.

The project is important because a useful model must solve two related but different tasks. First, it must rank fires by urgency so response teams can prioritize limited resources. Second, it must produce calibrated probabilities that can support decision thresholds. A fire predicted at 80 percent risk should genuinely be more urgent than one predicted at 20 percent risk.

---

## 2. Problem Statement

The main question is:

> Given early wildfire measurements from the first five hours, what is the probability that the fire will threaten an evacuation zone within 12, 24, 48, and 72 hours?

This is not a simple binary classification problem. A binary model can predict whether a fire eventually becomes dangerous, but it does not fully describe when the danger may happen. Emergency decisions depend strongly on timing. A fire likely to threaten a zone in 12 hours needs different action than a fire likely to threaten a zone in 72 hours.

The problem is also right-censored. Some fires do not reach an evacuation-zone centroid within the observation window. For those fires, the exact future time to threat is unknown; we only know that the event was not observed during the 72-hour period. This affects how training labels and evaluation metrics must be constructed.

The required submission contains four probability columns:

| Column | Meaning |
|---|---|
| `prob_12h` | Probability of threat within 12 hours |
| `prob_24h` | Probability of threat within 24 hours |
| `prob_48h` | Probability of threat within 48 hours |
| `prob_72h` | Probability of threat within 72 hours |

The probabilities should be monotonic. The probability of threat by 72 hours should not be lower than the probability of threat by 48 hours because the 72-hour window includes the earlier time window.

---

## 3. Objectives

The project objectives are:

1. Understand the wildfire survival-analysis task and right-censored target structure.
2. Inspect and explain the dataset columns in simple language.
3. Prepare a clean, reproducible notebook for multi-horizon probability prediction.
4. Construct horizon-specific labels that respect censoring.
5. Train baseline models that balance stability and nonlinear pattern detection.
6. Evaluate model quality using ranking and calibration metrics.
7. Generate a valid Kaggle submission file.
8. Improve project documentation, including README, dataset explanation, presentation materials, and Word report.

---

## 4. Dataset Description

The project uses four primary CSV files:

| File | Description |
|---|---|
| `train.csv` | Training dataset with wildfire features, event times, and event indicator |
| `test.csv` | Test dataset with wildfire features only |
| `metaData.csv` | Metadata describing columns, feature categories, and target variables |
| `sample_submission.csv` | Required Kaggle submission format |

The training dataset contains 221 wildfire events. The test dataset contains 95 wildfire events. The model uses 34 predictive features. The training data includes two target columns:

- `time_to_hit_hours`: time from the end of the first five-hour feature window until the fire comes within 5 km of an evacuation-zone centroid, or the last observed time for censored rows.
- `event`: indicator where `1` means the fire hit within the 72-hour window and `0` means the fire was censored.

The training data contains 69 observed events out of 221 rows, equal to approximately 31.2 percent. This means most fires in the training set did not reach the threat definition during the observation window.

---

## 5. Feature Groups

The dataset features can be understood through several groups.

### 5.1 Measurement Coverage

These features describe how much early fire-perimeter information is available:

- `num_perimeters_0_5h`
- `dt_first_last_0_5h`
- `low_temporal_resolution_0_5h`

If only one perimeter is available, the model has less evidence about fire movement and growth.

### 5.2 Fire Growth

Growth features describe the size of the fire and how quickly it expands:

- `area_first_ha`
- `area_growth_abs_0_5h`
- `area_growth_rel_0_5h`
- `area_growth_rate_ha_per_h`
- `log1p_area_first`
- `log1p_growth`
- `log_area_ratio_0_5h`
- `relative_growth_0_5h`
- `radial_growth_m`
- `radial_growth_rate_m_per_h`

Fast-growing fires are often more dangerous, especially when they are also close to populated or evacuation areas.

### 5.3 Fire Movement

Centroid movement features describe how the center of the fire changes over time:

- `centroid_displacement_m`
- `centroid_speed_m_per_h`
- `spread_bearing_deg`
- `spread_bearing_sin`
- `spread_bearing_cos`

The sine and cosine versions of spread bearing help machine learning models understand direction without treating 359 degrees and 1 degree as far apart.

### 5.4 Distance To Evacuation Zone

Distance features describe the relationship between the fire and the nearest evacuation-zone centroid:

- `dist_min_ci_0_5h`
- `dist_std_ci_0_5h`
- `dist_change_ci_0_5h`
- `dist_slope_ci_0_5h`
- `closing_speed_m_per_h`
- `closing_speed_abs_m_per_h`
- `projected_advance_m`
- `dist_accel_m_per_h2`
- `dist_fit_r2_0_5h`

These are highly relevant because a fire that is already close and moving closer is more likely to threaten an evacuation zone soon.

### 5.5 Directional Alignment

Directionality features describe whether the fire is moving toward or away from the evacuation zone:

- `alignment_cos`
- `alignment_abs`
- `cross_track_component`
- `along_track_speed`

Positive alignment and high along-track speed are important signs that the fire is moving in a dangerous direction.

### 5.6 Time Metadata

Time metadata features are:

- `event_start_hour`
- `event_start_dayofweek`
- `event_start_month`

These may indirectly capture environmental conditions, seasonal fire behavior, or response differences.

---

## 6. Methodology

The improved workflow is implemented in `TensorTitans_Improved.ipynb`.

### 6.1 Data Loading

The notebook loads `train.csv`, `test.csv`, `metaData.csv`, and `sample_submission.csv`. It verifies that the feature columns in the test set are present in the training set and that the submission columns match the expected format.

### 6.2 Horizon-Specific Label Construction

For each horizon, the notebook creates a binary target:

- label `1`: the fire event is observed by that horizon;
- label `0`: the fire is known not to have occurred by that horizon;
- excluded: the row is censored before that horizon, so its status is unknown.

The terminal 72-hour horizon is handled as a full-window event classification because non-event rows are known not to have hit within the competition window.

This is a major improvement over using the same binary `event` probability for every horizon.

### 6.3 Model Choice

The improved notebook uses a blended baseline:

1. **Regularized logistic regression**
   - stable on small datasets;
   - interpretable and less prone to overfitting;
   - uses median imputation, missing-value indicators, and standardization.

2. **Histogram gradient boosting classifier**
   - captures nonlinear relationships;
   - uses conservative settings to reduce overfitting;
   - complements the linear model.

The final probability is a weighted blend:

```text
0.70 x logistic regression + 0.30 x histogram gradient boosting
```

This blend is intentionally conservative because the dataset is small.

### 6.4 Cross-Validation

The notebook uses five-fold stratified cross-validation based on the final event indicator. For each fold and each horizon, the model is trained only on eligible rows and then predicts the held-out fold. This produces out-of-fold predictions for validation.

Out-of-fold validation is important because it tests the model on data that was not used for training. This gives a more realistic estimate of generalization performance than scoring on the training data.

### 6.5 Monotonic Probabilities

After prediction, probabilities are forced to be non-decreasing:

```text
prob_12h <= prob_24h <= prob_48h <= prob_72h
```

This matches the meaning of cumulative event probabilities.

---

## 7. Evaluation

The competition uses a hybrid metric combining ranking and calibration:

```text
Hybrid Score = 0.3 x C-index + 0.7 x (1 - Weighted Brier Score)
```

### 7.1 C-Index

C-index measures ranking quality. It checks whether fires that become threatening sooner receive higher risk scores than fires that become threatening later or remain censored.

### 7.2 Brier Score

Brier score measures probability calibration. A lower Brier score means the predicted probabilities are closer to the true outcomes.

The competition-weighted Brier score is:

```text
Weighted Brier = 0.3 x Brier@24h + 0.4 x Brier@48h + 0.3 x Brier@72h
```

The 48-hour horizon has the highest weight because it is operationally important: it gives useful warning time while still being close enough for emergency planning.

### 7.3 Reported Result

The team recorded a Kaggle hybrid score of 0.82691. This indicates useful performance in ranking wildfire threat urgency and producing probability estimates for the required horizons.

---

## 8. Results And Discussion

The original project notebook showed a baseline workflow for data loading, exploratory analysis, model training, and submission creation. However, the original modeling approach was closer to binary classification and could assign the same probability to all horizons. That weakens the survival-analysis interpretation because it does not fully model timing.

The improved notebook addresses this limitation by training separate horizon-specific models and using censor-aware validation. This makes the workflow more aligned with the competition objective. It also prevents logically inconsistent predictions by enforcing monotonic probabilities.

Important practical signals in the dataset include distance to evacuation-zone centroid, closing speed, fire growth rate, centroid movement, and directional alignment. A fire that is growing quickly, moving toward an evacuation zone, and already close to that zone should receive higher threat probability than a distant fire moving away.

Because the training dataset has only 221 rows, model complexity must be controlled carefully. A very complex neural network or large ensemble may overfit the training data. For that reason, the improved baseline uses regularized logistic regression as the main model and adds a small nonlinear component through histogram gradient boosting.

---

## 9. Project Deliverables

The cleaned project now contains:

| Deliverable | File |
|---|---|
| Improved modeling notebook | `TensorTitans_Improved.ipynb` |
| Original notebook for comparison | `TensorTitans.ipynb` |
| Dataset explanation | `Dataset_explanation.md` |
| Full report source | `Project_Report.md` |
| Full Word report | `TensorTitans_Project_Report.docx` |
| README documentation | `ReadMe.md` |
| Presentation slides | `Slide_Presentation.md` |
| Presentation script | `Presentation_Script.md` |
| Leaderboard image | `Leaderboard.png` |

Unnecessary clutter such as the empty text file, duplicate backup notebook, and generated submission CSV was removed from the repository.

---

## 10. Limitations

The current improved notebook is still a baseline. It does not yet use dedicated survival-analysis libraries such as `scikit-survival`. It also does not perform advanced probability calibration or repeated cross-validation. Because the dataset is small, validation scores can vary depending on fold splits.

Another limitation is that some feature descriptions in `metaData.csv` are high-level rather than fully detailed. The project therefore explains feature groups based on available metadata and column names.

---

## 11. Recommendations For Future Work

Future improvements should include:

1. Survival-specific models such as Coxnet, Random Survival Forest, and Gradient Boosting Survival Analysis.
2. Probability calibration using Platt scaling or isotonic regression inside cross-validation folds.
3. Repeated cross-validation for more stable validation estimates.
4. Feature selection to reduce noise and overfitting.
5. Model ensembling across survival models and horizon-specific classifiers.
6. Error analysis on fires that are close to evacuation zones but do not become events.
7. Better visual explanations of fire movement, distance trends, and directional alignment.

---

## 12. Conclusion

This project demonstrates a complete machine learning workflow for wildfire threat prediction using early fire behavior data. The task is meaningful because emergency managers need timely and calibrated forecasts to prioritize response actions. The dataset is challenging because it combines multi-horizon prediction with right-censored survival outcomes.

Team Tensor Titans developed the project from dataset inspection through model training, evaluation, submission formatting, and documentation. The cleaned repository now includes an improved censor-aware notebook, clearer README, dataset explanation, presentation materials, and this full Word report.

The most important technical lesson is that wildfire threat forecasting should account for timing. A model should not simply predict whether a fire ever becomes dangerous; it should estimate how soon that danger may occur. The improved workflow better reflects this by producing separate, monotonic probabilities for 12h, 24h, 48h, and 72h.

---

## 13. References

1. WiDS Worldwide Global Datathon 2026. https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26
2. Scikit-learn Documentation. https://scikit-learn.org/
3. PyTorch Documentation. https://pytorch.org/docs/
4. Project files: `train.csv`, `test.csv`, `metaData.csv`, `sample_submission.csv`, and `TensorTitans_Improved.ipynb`.

---

## Appendix A: Key Terms

**Fire perimeter:** the estimated boundary around the burning area.  
**Fire centroid:** the approximate center point of the fire perimeter.  
**Evacuation-zone centroid:** the center point of an evacuation zone.  
**Right censoring:** a situation where the event is not observed during the study window, so the exact future event time is unknown.  
**C-index:** a ranking metric for survival predictions.  
**Brier score:** a probability calibration error metric.  
**Monotonic probability:** a probability sequence that does not decrease as the time horizon becomes longer.
