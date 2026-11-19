# Practice Notebooks

Session and assignment notebooks from the DEPI / Udemy machine-learning practice, renamed after the dataset each one uses.
They show the step-by-step learning path (data preprocessing, visualization, encoding, scaling, outliers, imbalanced data, first models and model comparisons).

> The polished, complete versions of the NTI tasks (all models, GridSearchCV, full evaluation) are in the `Task01` – `Task04` folders and `Final_Project`.

| Notebook | Dataset | Topic | Original name |
|---|---|---|---|
| [Adult_Census_Income](Adult_Census_Income.ipynb) | Adult Census Income | preprocessing, SMOTE, Logistic Regression & KNN | Session8 |
| [Ames_Housing](Ames_Housing.ipynb) | Ames Housing | house price regression (Linear, Ridge, Decision Tree), GridSearchCV | Ames_Housing |
| [Bank_Marketing](Bank_Marketing.ipynb) | Bank Marketing (`bank.csv`) | term-deposit classification (Logistic Regression, SVM, Decision Tree), SMOTE, GridSearchCV | Assignment2 |
| [Breast_Cancer_Wisconsin](Breast_Cancer_Wisconsin.ipynb) | Breast Cancer Wisconsin (Diagnostic) | neural network (Keras) | Deep |
| [Canada_Immigration](Canada_Immigration.ipynb) | Immigration to Canada 1980 – 2013 | data visualization | Session3 |
| [Cars_Data](Cars_Data.ipynb) | Cars data | data preprocessing (missing values, encoding) | Session5 |
| [Customer_Segmentation](Customer_Segmentation.ipynb) | Customer Segmentation | preprocessing + Logistic Regression | Assignment |
| [Data_Science_Salaries](Data_Science_Salaries.ipynb) | Data science job salaries | data preprocessing & visualization | Session4 |
| [Diabetes_Patients](Diabetes_Patients.ipynb) | Diabetes patients (10,000) | classification with 6 models, GridSearchCV | Session10 |
| [Diabetes_Risk](Diabetes_Risk.ipynb) | Diabetes risk | classification with 6 models, outliers, GridSearchCV | Diabetes_Risk |
| [Employee_Attrition](Employee_Attrition.ipynb) | Employee attrition | classification with 6 models, outliers, GridSearchCV | Attrition |
| [House_Rent](House_Rent.ipynb) | House Rent Prediction | preprocessing + Linear Regression | Assignment_3 |
| [House_Rent_and_Stroke_Decision_Tree](House_Rent_and_Stroke_Decision_Tree.ipynb) | House Rent + Stroke | Decision Tree (regression & classification) | Decision Tree |
| [Laptop_Prices](Laptop_Prices.ipynb) | Laptop prices | regression, model comparison, ensembles | Laptop_csv |
| [SF_Salaries](SF_Salaries.ipynb) | San Francisco salaries | pandas data analysis | Session2 |
| [Stroke_Prediction](Stroke_Prediction.ipynb) | Stroke Prediction | preprocessing + imbalanced data (SMOTE, class weights) | Assignment_2 |
| [Student_Performance](Student_Performance.ipynb) | Student Performance | Linear Regression / Ridge | Session6 |
| [Telecom_Churn](Telecom_Churn.ipynb) | Telecom churn | churn classification with 6 models, GridSearchCV | Telecom_Churn |
| [WiDS_Datathon](WiDS_Datathon.ipynb) | WiDS Datathon (building energy) | data preprocessing + first models | Session7 |

**Note:** these notebooks read their CSV files from local paths on the author's computer (e.g. `/Users/.../NTI/`), so the datasets are not included here.
To re-run one, download the dataset and change the path in the `pd.read_csv(...)` cell.
