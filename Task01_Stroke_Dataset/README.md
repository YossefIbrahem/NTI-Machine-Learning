# Task 01 – Stroke Prediction (Binary Classification)

Predict whether a patient is likely to have a **stroke** from age, health conditions and lifestyle.

- **Dataset:** [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (`dataset/healthcare-dataset-stroke-data.csv`) – 5,110 patients, only **4.9 % with stroke** (imbalanced)
- **Notebook:** [`notebook.ipynb`](notebook.ipynb)
- **Main metric:** ROC-AUC (plus Recall, Precision, F1, PR curve) – accuracy is misleading on a 95 / 5 split

## Steps
1. **Cleaning** – dropped `id` and the single `gender = Other` row; `bmi` (201 missing) imputed with the median inside the pipeline; no duplicates.
2. **EDA & statistics** – age, glucose, BMI by stroke; stroke rate per group; t-tests and chi-square tests.
3. **Outliers** – high glucose values kept (real diabetics, stroke rate 13.4 % vs 4.9 %).
4. **Feature engineering** – `High_Glucose` (≥ 140) and `Senior` (age ≥ 60) flags.
5. **Feature selection** – `gender` (p = 0.56) and `Residence_type` (p = 0.30) dropped (not significant).
6. **Stratified train/test split** (80 / 20).
7. **Pipeline + ColumnTransformer + SMOTE** – SMOTE inside an `imblearn` pipeline so only the training folds are oversampled (no leakage).
8. **11 models** compared with stratified 5-fold CV → **GridSearchCV** on the top 3 → final test evaluation.
9. **Gradio section** at the end of the notebook – a small app to try the model on a new patient (runs inside the notebook).

## Results
| Model (CV, top 5) | ROC-AUC | Recall | F1 |
|---|---|---|---|
| **Logistic Regression** | **0.839** | 0.784 | 0.229 |
| LightGBM | 0.820 | 0.171 | 0.180 |
| Random Forest | 0.819 | 0.156 | 0.149 |
| AdaBoost | 0.817 | 0.714 | 0.229 |
| Gradient Boosting | 0.805 | 0.372 | 0.205 |

**GridSearchCV:** Logistic Regression `C = 0.1` (0.839 → 0.840), Random Forest `min_samples_leaf = 5` (0.819 → 0.829).

**Final model – Logistic Regression (test set):** ROC-AUC **0.839**, Recall **0.82** (41 of 50 stroke patients caught), Precision 0.14, AP 0.26, Accuracy 0.75.

- Age is by far the most important feature, then the High_Glucose and Senior flags.
- Many false alarms are accepted on purpose: in screening, a missed stroke is worse than an extra check-up (threshold trade-off shown in the notebook).

## Run
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm gradio
```
Open `notebook.ipynb` and run all cells (≈ 1 minute).
> **Python 3.9 only:** also run `pip install "huggingface_hub<1.0"` – gradio 4.44 (the last version for Python 3.9) does not work with huggingface_hub 1.x (`ImportError: cannot import name 'HfFolder'`).

