# NTI – Machine Learning for Data Analysis

Tasks and final project for the **Machine Learning for Data Analysis** track at NTI (National Telecommunication Institute).
Each folder has runnable code, its dataset (or a link to it) and a README that explains the idea, the steps and the results.

## Final Project – Egypt House Price Prediction 🏠
Predicting the price (EGP) of a property in Egypt from 27k+ real listings.

- Data cleaning, EDA, outlier handling, feature engineering, Pipeline + ColumnTransformer
- 13 regression models compared with 5-fold cross-validation, GridSearchCV on the top 3
- Best model: **SVR** – test **R² = 0.67** (log price), median error ≈ 25 %
- Deployed as a **Gradio** app

📁 [Final_Project](Final_Project) · 🚀 Live app: `ADD_YOUR_HUGGING_FACE_LINK_HERE`

## Tasks
Every task follows the same workflow: data understanding → cleaning → EDA & statistical tests → outliers → feature engineering / selection →
train/test split → Pipeline + ColumnTransformer → all suitable models with cross-validation → comparison table → GridSearchCV (top 3) → final test evaluation.

| # | Task | Type | Models | Best model | Test result | Live app |
|---|---|---|---|---|---|---|
| 01 | [Stroke Prediction](Task01_Stroke_Dataset) | Binary classification (imbalanced, SMOTE) | 11 | Logistic Regression | ROC-AUC 0.84, Recall 0.82 | – |
| 02 | [House Rent](Task02_House_Rent) | Regression | 13 | Gradient Boosting | R² 0.83 (log rent) | – |
| 03 | [Student Performance](Task03_Student_Performance) | Regression | 13 | Ridge | R² 0.989, MAE 1.65 | 🚀 `ADD_LINK` |
| 04 | [Customer Segmentation](Task04_Customer_Segmentation) | Multiclass classification (4 segments) | 11 | LightGBM | Accuracy 0.54, ROC-AUC 0.80 | – |

## Repository Structure
```
NTI-Machine-Learning/
├── Task01_Stroke_Dataset/          # dataset/, notebook.ipynb, README.md
├── Task02_House_Rent/              # dataset/, notebook.ipynb, README.md
├── Task03_Student_Performance/     # dataset/, notebook.ipynb, README.md, app.py, requirements.txt, models/
├── Task04_Customer_Segmentation/   # dataset/, notebook.ipynb, README.md
├── Final_Project/
│   ├── dataset/  notebooks/  models/  results/
│   ├── app.py  requirements.txt  README.md
└── README.md
```
