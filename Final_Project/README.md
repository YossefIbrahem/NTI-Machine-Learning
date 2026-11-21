# Egypt House Price Prediction 🏠

Predicting the price (EGP) of a property in Egypt from its description, using a full machine-learning workflow:
data cleaning → EDA → feature engineering → Pipeline / ColumnTransformer → 13 regression models → GridSearchCV → test evaluation → deployment.

**Live app:** https://huggingface.co/spaces/YoussefIbrahem0/egypt-house-price-prediction

---

## 1. Problem Statement
Buyers and sellers need a fair price estimate before they negotiate. Given the type, size, rooms, location, compound,
finishing, payment and delivery of a property, the model predicts its **price in EGP**.

- **Task type:** supervised **regression** (target = `Price`)
- **Main metric:** R² (plus MAE, MSE, RMSE)

## 2. Dataset
`dataset/egypt_house_prices.csv` – 27,361 listings, 11 features + target (dataset provided by the NTI instructor).

| Column | Meaning |
|---|---|
| Type | Apartment, Chalet, Stand Alone Villa, Town House, Twin House, Duplex, Penthouse, Studio |
| **Price** | **Target** – price in EGP |
| Bedrooms / Bathrooms | number of rooms (`10+` = 10 or more) |
| Area | size in m² |
| Furnished | Yes / No / Unknown |
| Level | floor number, Ground, Highest or Unknown |
| Compound | compound name, Unknown or Not in Compound |
| Payment_Option | Cash / Installment / Cash or Installment |
| Delivery_Date | Ready to move, soon, within 6 months, 2022 … 2027 |
| Delivery_Term | Finished / Semi Finished / Not Finished / Core & Shell |
| City | city or area (180+ values) |

## 3. Project Structure
```
Final_Project/
├── dataset/egypt_house_prices.csv
├── notebooks/egypt_house_prices.ipynb   # full analysis + modelling
├── models/house_price_model.pkl         # trained Pipeline (preprocessing + best model)
├── models/choices.json                  # dropdown values used by the app
├── results/                             # comparison tables and plots
├── static_app/                          # live web app (index.html, model.js, model.json, export_model.py)
├── app.py                               # Gradio app (run locally)
├── requirements.txt
├── linkedin_post.md                     # draft of the LinkedIn post
└── README.md
```

## 4. Steps
1. **Data understanding** – every column was stored as text, `Unknown` was used instead of empty values, the target is `Price` → regression.
2. **Data cleaning** – stripped spaces, merged duplicated labels (`Standalone Villa` → `Stand Alone Villa`, `Twin house` → `Twin House`),
   converted Price / Area / Bedrooms / Bathrooms to numbers (`10+` → 10, `Unknown` → NaN), dropped 96 rows without a price and 1,862 duplicates.
3. **EDA & statistics** – distributions, skewness (price skew = 6.8 → log target), price by type / city / rooms / finishing / payment, Spearman correlations.
4. **Outliers** – judged by **price per m²** (a 50M villa is real, a 100k chalet is not). IQR limits learned **from the training set only**
   (1,030 – 171,739 EGP/m²), 200 train rows and 38 test rows removed.
5. **Feature engineering** – `Log_Area` and `Area_per_Bedroom` (price per m² is *not* used as it contains the target).
6. **Feature selection** – rare categories of Compound / City grouped with `OneHotEncoder(min_frequency=10)` → 322 columns.
7. **Train/test split** – 80 / 20, done before any learned preprocessing.
8. **Pipeline + ColumnTransformer** – median imputer + StandardScaler for numbers, `Unknown` imputer + OneHotEncoder for categories → no data leakage.
9. **13 models** compared with 5-fold cross-validation on the training data.
10. **GridSearchCV** on the top 3 models, best model chosen by CV score (the test set is used only once at the end).
11. **Evaluation** on the test set – MAE, MSE, RMSE, R², actual-vs-predicted, residuals, error by type, permutation importance.
12. **Deployment** – the saved Pipeline is exported to a web app on Hugging Face Spaces (`static_app/`), and it is also served with Gradio (`app.py`, and the Gradio section at the end of the notebook).

## 5. Results

### Model comparison (5-fold CV on training data, log price)
| Model | Train R² | CV R² | CV MAE | CV RMSE |
|---|---|---|---|---|
| **SVR** | 0.730 | **0.675** | 0.455 | 0.684 |
| XGBoost | 0.744 | 0.668 | 0.502 | 0.692 |
| Random Forest | 0.942 | 0.665 | 0.475 | 0.695 |
| LightGBM | 0.705 | 0.663 | 0.509 | 0.697 |
| Extra Trees | 0.985 | 0.649 | 0.464 | 0.711 |
| KNN | 0.754 | 0.628 | 0.513 | 0.732 |
| Ridge | 0.623 | 0.612 | 0.568 | 0.748 |
| Linear Regression | 0.623 | 0.612 | 0.568 | 0.748 |
| Gradient Boosting | 0.621 | 0.608 | 0.577 | 0.751 |
| ElasticNet | 0.605 | 0.599 | 0.584 | 0.760 |
| Lasso | 0.595 | 0.590 | 0.594 | 0.768 |
| AdaBoost | 0.432 | 0.426 | 0.777 | 0.909 |
| Decision Tree | 0.985 | 0.396 | 0.599 | 0.933 |

- Linear models **underfit** (train ≈ CV ≈ 0.61); the single Decision Tree **overfits** (0.98 → 0.40).
- Non-linear models (SVR, XGBoost, Random Forest, LightGBM) are the best.

### GridSearchCV (top 3)
| Model | CV R² before | CV R² after | best_params_ |
|---|---|---|---|
| **SVR** | 0.675 | **0.684** | C = 3, gamma = 'scale' |
| XGBoost | 0.668 | 0.681 | n_estimators = 600, max_depth = 8, learning_rate = 0.1 |
| Random Forest | 0.665 | 0.673 | n_estimators = 200, max_features = 0.3, min_samples_leaf = 1 |

### Final test evaluation – SVR
| | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Before tuning (log price) | 0.454 | 0.475 | 0.690 | 0.667 |
| **After tuning (log price)** | **0.446** | **0.467** | **0.683** | **0.673** |
| After tuning (EGP) | 1.69M | 2.46e13 | 4.96M | 0.544 |

- Median error **25 %**, 43 % of predictions within ±20 %, 74 % within ±50 %.
- Best predicted: Twin House (18 % median error) and Apartment (20 %); hardest: Chalet (35 %).
- Most important features: **Compound, Log_Area, City** → location and size drive the price.

Plots are in [`results/`](results): model comparison, actual vs predicted, residuals, feature importance.

## 6. Run locally
```bash
pip install -r requirements.txt
python app.py              # opens the app at http://127.0.0.1:7860
```
To re-train, open `notebooks/egypt_house_prices.ipynb` and run all cells (about 20 minutes, mostly SVR and the grid search).
The notebook saves the model to `models/` and the tables / plots to `results/`.

## 7. Deployment (Hugging Face Spaces – free static Space)
**Live app:** https://huggingface.co/spaces/YoussefIbrahem0/egypt-house-price-prediction

Hugging Face now needs a PRO subscription to host Gradio Spaces, while static Spaces stay free. So the trained model runs directly in the browser:

1. `static_app/export_model.py` reads `models/house_price_model.pkl` and saves everything the Pipeline needs to `static_app/model.json`:
   the imputer and scaler values, the one-hot categories (including the "infrequent" group for rare compounds and cities)
   and the SVR model (15,271 support vectors, dual coefficients, gamma, intercept).
2. `static_app/model.js` repeats the Pipeline step by step – feature engineering → scaling → one-hot → RBF-kernel SVR → `expm1` of the log price.
3. `static_app/index.html` is the form; it predicts instantly as you type.

The JavaScript predictions were checked against `model.predict` on 3,011 inputs (every compound and city, unseen values, the app examples):
the largest difference was 0.000007 EGP, so the web app gives **exactly** the same prices as the Python model.

To update the live app after re-training: run `python static_app/export_model.py`, then upload `index.html`, `model.js` and `model.json` to the Space.

## 8. Conclusions & Recommendations
- Location (Compound, City) and size (Area, rooms) explain most of the price; non-linear models clearly beat linear ones.
- The tuned SVR reaches **R² = 0.67 on the test set** (log price) with a typical error of about 25 % – good as a first estimate, not as an exact valuation.
- All strong models stop around R² ≈ 0.68: the data has no land area, view, building age, exact location or listing date, and some prices look like down-payments.
  Adding these fields, separating coastal from city properties, and retraining regularly (inflation) would improve the model the most.
