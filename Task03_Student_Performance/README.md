# Task 03 – Student Performance Prediction (Regression) 🚀 Deployed

Predict a student's **Performance Index (10 – 100)** from study hours, previous scores, sleep, practice papers and extracurricular activities.

- **Dataset:** `dataset/Student_Performance.csv` – 10,000 students, 5 features
- **Notebook:** [`notebook.ipynb`](notebook.ipynb)
- **App:** [`app.py`](app.py) (Gradio) – **Live app:** `ADD_YOUR_HUGGING_FACE_LINK_HERE`
- **Main metric:** R², plus MAE, MSE, RMSE

## Steps
1. **Cleaning** – 132 duplicates removed (before the split); missing `Extracurricular Activities` (1,572) and `Sample Question Papers Practiced` (2,856) imputed inside the pipeline.
2. **EDA & statistics** – target is normal; strong straight-line relation with Previous Scores (r = 0.92) and Hours Studied (r = 0.37); t-test and Pearson tests.
3. **Outliers** – none by the IQR rule (all values in normal ranges).
4. **Feature engineering / selection** – not needed: clean, independent, linear features; all 5 are significant (p < 0.05) and kept.
5. **Train/test split** (80 / 20).
6. **Pipeline + ColumnTransformer** – imputers, StandardScaler, OneHotEncoder (`drop='if_binary'`).
7. **13 regression models** compared with 5-fold CV → **GridSearchCV** on the top 3 → final test evaluation → model saved for the app.
8. **Gradio section** at the end of the notebook – runs the app inside the notebook (the same app as `app.py`).

## Results
| Model (CV, top 6) | Train R² | CV R² | CV RMSE |
|---|---|---|---|
| **Ridge** | 0.9883 | **0.9883** | 2.075 |
| Linear Regression | 0.9883 | 0.9883 | 2.075 |
| Lasso | 0.9883 | 0.9883 | 2.075 |
| ElasticNet | 0.9883 | 0.9882 | 2.078 |
| Gradient Boosting | 0.9885 | 0.9874 | 2.151 |
| LightGBM | 0.9901 | 0.9872 | 2.164 |

Linear models win because the relations are straight lines; trees and boosting are close but slightly worse. Tuning gave no extra gain (no regularization needed).

**Final model – Ridge (test set):** R² **0.989**, MAE **1.65** points, RMSE 2.08; 94 % of predictions within ±4 points.

Coefficients: +2.86 points per study hour, +1.02 per previous-score point, +0.48 per sleep hour, +0.19 per practice paper, +0.48 for extracurricular activities.

## Run the app locally
```bash
pip install -r requirements.txt
python app.py              # opens http://127.0.0.1:7860
```
The app loads `models/student_model.pkl` – the full pipeline (imputers + scaler + encoder + Ridge) saved by the notebook – so the input is preprocessed exactly like in training.

## Deploy (Hugging Face Spaces – free)
1. Create a new Space at https://huggingface.co/new-space → SDK **Gradio**.
2. Upload `app.py`, `requirements.txt` and the `models/` folder.
3. When the build finishes, the Space link is the live deployment link – paste it above.

To re-train: `pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm gradio`, then run all cells of `notebook.ipynb` (≈ 30 seconds).

> **Python 3.9 only:** also run `pip install "huggingface_hub<1.0"` – gradio 4.44 (the last version for Python 3.9) does not work with huggingface_hub 1.x (`ImportError: cannot import name 'HfFolder'`).

