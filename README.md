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
| Task | Folder | Live deployment |
|---|---|---|
| *(add each task folder here)* | | |

## Repository Structure
```
NTI-Machine-Learning/
├── TaskXX_.../            # one folder per task: dataset/, notebook.ipynb, README.md
├── Final_Project/
│   ├── dataset/  notebooks/  models/  results/
│   ├── app.py  requirements.txt  README.md
└── README.md
```
