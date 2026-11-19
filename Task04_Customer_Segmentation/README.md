# Task 04 – Customer Segmentation (Multiclass Classification)

Predict the **segment (A, B, C, D)** of a new customer of an automobile company, so the right marketing plan is used.

- **Dataset:** [Kaggle – Customer Segmentation](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation) (`dataset/Train.csv`, `dataset/Test.csv`) – 8,068 customers, 4 almost balanced segments
- **Notebook:** [`notebook.ipynb`](notebook.ipynb)
- **Main metric:** Accuracy, plus macro Precision / Recall / F1, ROC-AUC and AP (one-vs-rest, macro average)

## Steps
1. **Checking `Test.csv`** – its IDs overlap with `Train.csv` but belong to different people, and every model gets only ~34 % on it (vs ~53 % in CV) → its labels don't follow the same pattern.
   The official test set is a **stratified 80 / 20 split of `Train.csv`**; `Test.csv` is reported only as a side note.
2. **Cleaning** – dropped `ID`; missing values (Work_Experience 10 %, Family_Size 4 %, …) imputed inside the pipeline; 417 identical profiles kept (different real customers).
3. **EDA & statistics** – segment profiles (D = young, single, low spending; C = older, married, graduated); chi-square and ANOVA tests (all features significant).
4. **Outliers** – long work experience and large families are real values → kept.
5. **Encoding** – `Spending_Score` with **OrdinalEncoder** (Low < Average < High), other categories with OneHotEncoder.
6. **Pipeline + ColumnTransformer** – no data leakage.
7. **11 classification models** compared with stratified 5-fold CV → **GridSearchCV** on the top 3 → final test evaluation.
8. **Gradio section** at the end of the notebook – a small app that predicts the segment of a new customer and shows the probability of all 4 segments.

## Results
| Model (CV, top 5) | Train Acc. | CV Accuracy | F1 (macro) | ROC-AUC (ovr) |
|---|---|---|---|---|
| Gradient Boosting | 0.593 | **0.531** | 0.517 | 0.781 |
| SVM | 0.579 | 0.525 | 0.510 | 0.774 |
| LightGBM | 0.724 | 0.520 | 0.506 | 0.772 |
| XGBoost | 0.802 | 0.503 | 0.492 | 0.762 |
| Logistic Regression | 0.508 | 0.503 | 0.474 | 0.751 |

Decision Tree, Random Forest and Extra Trees overfit badly (0.96 train → 0.44 – 0.48 CV).

**GridSearchCV:** LightGBM 0.520 → **0.536** (`learning_rate = 0.03`, 100 trees, 15 leaves) became the best; Gradient Boosting and SVM were already best with their defaults.

**Final model – LightGBM (test split):** Accuracy **0.544** (random = 0.25), macro F1 0.524, ROC-AUC (ovr, macro) **0.797**, macro AP 0.569.

- Segment **D** is the easiest (recall 0.79, AUC 0.89); **A and B** overlap a lot and are the hardest.
- Most important features: **Age, Profession, Spending_Score**.

## Run
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm gradio
```
Open `notebook.ipynb` and run all cells (≈ 2 minutes).
> **Python 3.9 only:** also run `pip install "huggingface_hub<1.0"` – gradio 4.44 (the last version for Python 3.9) does not work with huggingface_hub 1.x (`ImportError: cannot import name 'HfFolder'`).

