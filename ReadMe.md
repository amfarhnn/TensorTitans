# Tensor Titans

## WiDS Worldwide Global Datathon 2026 - Wildfire Survival Analysis

Tensor Titans is a machine learning project for the WiDS Worldwide Global Datathon 2026. The task is to predict wildfire threat probabilities across multiple time horizons using only early fire signals from the first five hours after initial observation.

The practical question is:

> Given the first five hours of wildfire behavior, how likely is the fire to come within 5 km of an evacuation-zone centroid within 12, 24, 48, or 72 hours?

The project uses right-censored survival data, so the model must consider both whether a fire reaches the threat zone and how soon that happens.

## Team Members

| Member | ID |
|---|---|
| Amir Farhan bin Ghaffar | 2115617 |
| Muhammad Irsyad Ilham bin Azizan | 2217555 |
| Muhammad Amin bin Mohamad Rizal | 2217535 |

**Project Track:** Track 1 - Kaggle Competition  
**Competition:** [WiDS Worldwide Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)  
**Reported leaderboard score:** 0.82691 hybrid score

## Repository Contents

| File | Purpose |
|---|---|
| `TensorTitans_Improved.ipynb` | Main improved notebook with censor-aware multi-horizon modeling and submission generation. |
| `TensorTitans.ipynb` | Original project notebook kept for comparison/history. |
| `wids-datathon-2026-winning-solution-btt-mapping.ipynb` | Reference notebook for advanced survival modeling ideas. |
| `train.csv` | Training data with features, event times, and event indicator. |
| `test.csv` | Test data used to generate Kaggle probabilities. |
| `metaData.csv` | Column descriptions and feature categories. |
| `sample_submission.csv` | Required Kaggle submission format. |
| `Dataset_explanation.md` | Plain-language explanation of every dataset column. |
| `Slide_Presentation.md` | Marp slide deck source. |
| `Presentation_Script.md` | Five-minute presentation script. |
| `Leaderboard.png` | Leaderboard screenshot. |

Some report-formatting files are intentionally kept local and excluded from GitHub:

| Local file | Purpose |
|---|---|
| `Project_Report.md` | Full report source used for document preparation. |
| `TensorTitans_Project_Report.docx` | Microsoft Word report generated from the report source. |
| `TensorTitans_Project_Report.pdf` | PDF export of the report, if generated locally. |
| `Follow_this_format_and_style.docx` | Reference template used to style the Word report. |

## Dataset Summary

- **Training rows:** 221 wildfire events
- **Test rows:** 95 wildfire events
- **Input features:** 34 engineered wildfire features
- **Target columns:** `time_to_hit_hours`, `event`
- **Observed events:** 69 out of 221 training fires, approximately 31.2%
- **Prediction horizons:** 12h, 24h, 48h, 72h
- **Threat definition:** fire comes within 5 km of an evacuation-zone centroid

The features describe early wildfire growth, movement, distance to evacuation zones, directional alignment, and event start time. All predictive features are computed from the first five hours after initial observation.

## Modeling Approach

The improved notebook uses a conservative, reproducible baseline suitable for the small training set:

1. Load and inspect the wildfire data.
2. Build horizon-specific labels for 12h, 24h, 48h, and 72h.
3. Handle censoring by excluding rows whose status is unknown at a horizon.
4. Train a blended model:
   - regularized logistic regression for stable probabilities,
   - histogram gradient boosting for limited nonlinear signal.
5. Evaluate with out-of-fold validation using:
   - C-index for ranking quality,
   - censor-aware Brier score for probability calibration,
   - hybrid score combining both objectives.
6. Enforce monotonic probabilities so later horizons are never lower than earlier horizons.
7. Generate `submission_improved.csv`.

## Evaluation Metric

The competition uses a hybrid metric:

```text
Hybrid Score = 0.3 x C-index + 0.7 x (1 - Weighted Brier Score)
```

The weighted Brier score emphasizes the operationally important horizons:

```text
Weighted Brier = 0.3 x Brier@24h + 0.4 x Brier@48h + 0.3 x Brier@72h
```

This means a good model must do two things:

- rank dangerous fires earlier than less dangerous fires;
- output calibrated probabilities that can be trusted for decision thresholds.

## How To Run

### Option 1: Google Colab

Open `TensorTitans_Improved.ipynb` in Colab, upload the CSV files if needed, and run all cells from top to bottom.

### Option 2: Local Python

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Then open and run:

```text
TensorTitans_Improved.ipynb
```

The notebook will create:

```text
submission_improved.csv
```

Generated submission files are ignored by Git so they do not accidentally overwrite or clutter the repository.

## Report And Presentation

The report and presentation materials are maintained separately from the modeling notebook:

- `Dataset_explanation.md` explains the wildfire dataset in beginner-friendly language.
- `Slide_Presentation.md` contains the presentation slide source.
- `Presentation_Script.md` contains a five-minute speaking script.
- `TensorTitans_Project_Report.docx` and `TensorTitans_Project_Report.pdf` are local report exports and are not pushed to GitHub.

## Key Learning Points

- Wildfire threat prediction is not only a binary classification problem; timing matters.
- Right-censored observations must be handled carefully because some fires are not observed long enough to know their status at every horizon.
- Distance, closing speed, direction of spread, and alignment with evacuation zones are important practical signals.
- Multi-horizon probabilities should be monotonic: risk by 72h should be at least as high as risk by 48h.

## Future Improvements

- Add survival-specific models such as Coxnet, Random Survival Forest, or Gradient Boosting Survival Analysis.
- Calibrate probabilities using fold-safe Platt scaling or isotonic regression.
- Use repeated cross-validation to reduce validation variance.
- Test compact feature subsets to avoid overfitting.
- Build an ensemble using both survival models and horizon-specific classifiers.

## References

- WiDS Worldwide Global Datathon 2026: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26
- Scikit-learn documentation: https://scikit-learn.org/
- PyTorch documentation: https://pytorch.org/docs/

## Status

Project cleaned and updated with an improved notebook, clearer documentation, and local report exports.
