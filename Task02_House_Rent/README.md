# Task 02 – House Rent Prediction (Regression)

Predict the **monthly rent (INR)** of a house / flat in 6 Indian cities from size, rooms, floor, city and furnishing.

- **Dataset:** [Kaggle – House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) (`dataset/House_Rent_Dataset.csv`) – 4,746 listings
- **Notebook:** [`notebook.ipynb`](notebook.ipynb)
- **Main metric:** R² on log(Rent), plus MAE, MSE, RMSE

## Steps
1. **Cleaning** – dropped 5 rows without rent; no duplicates; missing Size / Area Type / Bathroom imputed inside the pipeline.
2. **EDA & statistics** – rent is extremely skewed (skewness 21.4) → **log target**; rent by city, size, BHK, furnishing, point of contact; ANOVA tests.
3. **Feature engineering** – `Floor` text (`3 out of 5`) parsed into `Floor_Num` and `Total_Floors`; `Log_Size`, `Size_per_BHK`.
4. **Feature selection** – dropped `Area Locality` (2,235 unique values), `Posted On` (only 4 months) and the raw `Floor` text.
5. **Train/test split** (80 / 20), then **outliers** removed with the IQR rule on log(rent per sqft) – limits learned from the training data only.
6. **Pipeline + ColumnTransformer** – median / most-frequent imputers, StandardScaler, OneHotEncoder (`drop='first'` to avoid the dummy variable trap).
7. **13 regression models** compared with 5-fold CV → **GridSearchCV** on the top 3 → final test evaluation.
8. **Gradio section** at the end of the notebook – a small app to estimate the rent of a new flat (runs inside the notebook).

## Results
| Model (CV, top 5) | Train R² | CV R² | CV RMSE |
|---|---|---|---|
| LightGBM | 0.912 | **0.841** | 0.367 |
| Gradient Boosting | 0.870 | 0.840 | 0.368 |
| SVR | 0.876 | 0.839 | 0.369 |
| XGBoost | 0.960 | 0.830 | 0.379 |
| Random Forest | 0.973 | 0.829 | 0.381 |

Linear models reach 0.829 (close to the best, because log rent is almost linear in log size); a single Decision Tree overfits (0.996 → 0.700).

**GridSearchCV:** Gradient Boosting 0.840 → **0.845** (`learning_rate = 0.05`, `max_depth = 4`, 400 trees), LightGBM 0.841 → 0.844, SVR 0.838 → 0.842.

**Final model – Gradient Boosting (test set):** R² **0.83** (log rent), RMSE 0.38, median error **≈ 22 %**, 87 % of predictions within ±50 %.

- Most important features: **City**, Point of Contact, Size, Bathroom, BHK.
- Mumbai is the hardest city to predict (24 % median error), Bangalore the easiest (19.6 %).

## Run
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm gradio
```
Open `notebook.ipynb` and run all cells (≈ 1 minute).
> **Python 3.9 only:** also run `pip install "huggingface_hub<1.0"` – gradio 4.44 (the last version for Python 3.9) does not work with huggingface_hub 1.x (`ImportError: cannot import name 'HfFolder'`).

