# 5-Minute Presentation Script

## Team Tensor Titans - Wildfire Survival Analysis Challenge

### Time Plan
- Speaker 1: about 1 minute 40 seconds
- Speaker 2: about 1 minute 40 seconds
- Speaker 3: about 1 minute 40 seconds

---

## Speaker 1: Deliverable 1 and Deliverable 2

Our first deliverable was to understand the competition task and the survival analysis framework. The project is based on the WiDS Worldwide Global Datathon 2026, where the goal is to predict wildfire threat probability across multiple time horizons using early-incident signals from the first 5 hours after fire ignition.

From the dataset files, we used `train.csv` for model training with right-censored survival data, `test.csv` for multi-horizon prediction, `metaData.csv` to understand the wildfire features, and `sample_submission.csv` as the reference format for probability submissions.

Our first analysis showed that the dataset is a right-censored survival problem. Approximately 15-20% of fires threaten evacuation zones within 72 hours, while the rest are censored. Because of this survival structure and the need for both ranking and calibration, the competition uses a hybrid metric: 30% C-index for ranking quality plus 70% weighted Brier Score for probability calibration.

---

## Speaker 2: Deliverable 3 and Deliverable 4

Our second deliverable was exploratory data analysis. We inspected the dataset structure, examined wildfire perimeter dynamics, analyzed threat event timing, and studied the spatial relationships between fires and evacuation zones.

Based on that analysis, we prepared the data by standardizing numerical features using `StandardScaler` and handling missing values appropriately. We removed irrelevant identifier columns.

For the modeling deliverable, we built two baseline approaches. The first was a logistic regression pipeline for generating calibrated probabilities. The second was a feedforward neural network implemented in PyTorch with multiple dense layers and sigmoid outputs for each time horizon (12h, 24h, 48h, 72h).

We used a stratified train-validation split to evaluate the models. After training, we generated survival probability predictions and calculated both C-index for ranking and censor-aware Brier Scores for calibration.

---

## Speaker 3: Deliverable 5 and Final Result

Our final deliverable was the competition submission. Using the trained baseline survival models, we generated multi-horizon probability forecasts for the test set and formatted them according to the Kaggle submission template with probability columns for 12h, 24h, 48h, and 72h horizons.

The result of our submission was a hybrid score of 0.82691 on the leaderboard. This score reflects strong performance in both ranking fires by urgency and generating calibrated probability forecasts.

![Leaderboard Screenshot](Leaderboard.png)

In summary, we completed the full survival analysis workflow: understanding the right-censored data, exploring early-incident signals, preprocessing, building survival models, and submitting calibrated multi-horizon probability forecasts. For future improvement, we can implement advanced survival models like Cox proportional hazards or random survival forests, apply probability calibration techniques, and conduct more sophisticated feature engineering.

---

## Closing Line

That is our deliverable summary for the wildfire survival analysis project.