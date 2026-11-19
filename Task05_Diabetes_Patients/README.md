# Task 05 – Diabetes Prediction (Binary Classification) 🚀 Deployed

Predict whether a patient is **Diabetic** or **Non-Diabetic** from lab results, as a screening aid.

- **Dataset:** `dataset/diabetes_patients.csv` – 10,000 patients, 8 features (Age, Gender, Blood Sugar, Creatinine, BMI, Urea, Cholesterol, HbA1c), target `Class`, with missing values; **imbalanced** (91 % Diabetic)
- **Notebook:** [`notebook.ipynb`](notebook.ipynb)
- **App:** [`app.py`](app.py) (Gradio) – **Live app:** `ADD_YOUR_HUGGING_FACE_LINK_HERE`
- **Main metric:** F1 (macro) – both classes count equally – plus recall per class, ROC-AUC, PR curves

## Steps
1. **Cleaning** – dropped 15 rows without a class, the `Patient_ID` column and 56 almost-empty rows (5+ of 8 features missing); no duplicates left; other missing values imputed inside the pipeline.
2. **EDA & statistics** – distributions by class, t-tests and chi-square: only **HbA1c** and **Blood Sugar** differ between the classes.
   The scatter plot shows the rule: **Non-Diabetic ⇔ HbA1c ≤ 6.0 % and sugar ≤ 140 mg/dL**.
3. **Outliers** – none (all values within normal medical ranges).
4. **Feature engineering** – medical flags `High_HbA1c` (> 6.0 %) and `High_Sugar` (> 140 mg/dL).
5. **Stratified train/test split** (80 / 20).
6. **Feature selection on the training data** – Age, Gender, Creatinine, BMI, Urea, Cholesterol not significant → dropped (CV confirmed no loss).
7. **Imbalance** – SMOTE was tested and made the model worse (0.9885 → 0.983), so stratified CV + macro F1 are used instead.
8. **Pipeline + ColumnTransformer** (median imputer + StandardScaler).
9. **11 classification models** compared with stratified 5-fold CV → **GridSearchCV** on the top 3 → final test evaluation → error analysis → model saved for the app.
10. **Gradio section** at the end of the notebook – runs the app inside the notebook (the same app as `app.py`).

## Results
| Model (CV) | F1 (macro) | Recall Non-Diabetic | Recall Diabetic | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | **0.9885** | 0.959 | 1.000 | 0.998 |
| SVM | 0.9885 | 0.959 | 1.000 | 0.999 |
| AdaBoost | 0.9885 | 0.959 | 1.000 | 1.000 |
| Decision Tree | 0.9883 | 0.978 | 0.998 | 0.995 |
| Extra Trees | 0.9883 | 0.978 | 0.998 | 0.995 |
| LightGBM | 0.9882 | 0.972 | 0.999 | 1.000 |
| XGBoost | 0.9882 | 0.969 | 0.999 | 1.000 |
| Random Forest | 0.9878 | 0.971 | 0.999 | 0.998 |
| Gradient Boosting | 0.9877 | 0.965 | 0.999 | 1.000 |
| KNN | 0.9862 | 0.969 | 0.998 | 0.995 |
| Naive Bayes | 0.9184 | 0.960 | 0.972 | 0.998 |

**GridSearchCV:** the top 3 stay at 0.9885 – nothing left to tune; **Logistic Regression (C = 0.1)** is chosen as the simplest.

**Final model – Logistic Regression (test set):** accuracy **99.75 %**, F1 (macro) **0.992**, ROC-AUC **0.998**,
Diabetic recall **1.000** (no diabetic patient missed), Non-Diabetic recall 0.971.

- Only **5 mistakes out of 1,986**, all Non-Diabetic patients with a **missing blood sugar** value. With both lab values known the model is **100 % correct**.
- A depth-2 Decision Tree trained on the training data finds the same medical rule: HbA1c ≤ 6.05 and sugar ≤ 140.5 → Non-Diabetic.
- The data looks generated from this rule; real hospital data would be harder.

## Run the app locally
```bash
pip install -r requirements.txt
python app.py              # opens http://127.0.0.1:7860
```
The app asks only for **HbA1c** and **blood sugar** (the only useful features), adds the same two flags as the notebook, and loads
`models/diabetes_model.pkl` – the full pipeline (imputer + scaler + Logistic Regression).

## Deploy (Hugging Face Spaces – free)
1. Create a new Space at https://huggingface.co/new-space → SDK **Gradio**.
2. Upload `app.py`, `requirements.txt` and the `models/` folder.
3. When the build finishes, the Space link is the live deployment link – paste it above.

To re-train: `pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm`, then run all cells of `notebook.ipynb` (≈ 20 seconds).
