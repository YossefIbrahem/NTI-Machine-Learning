🏠 How much is a house in Egypt worth? I built a Machine Learning model to answer that.

As part of the Machine Learning for Data Analysis track at NTI (National Telecommunication Institute), I worked on predicting property prices in Egypt from 27,000+ real listings.

📊 The data
Type, area, bedrooms, bathrooms, compound, city, finishing, payment option and delivery date – all stored as text with many "Unknown" values.

🧹 Cleaning & preprocessing
• Converted text to numbers, merged duplicated labels, removed 1,862 duplicate rows
• Found prices that were really down-payments (a 120 m² chalet for 100,000 EGP!) and removed outliers using the IQR rule on price per m², learned from the training data only
• Predicted log(Price) because the price was very skewed (skewness 6.8)
• Built a Pipeline + ColumnTransformer (imputation, scaling, one-hot encoding) to avoid data leakage

🤖 Models
I compared 13 regression models with 5-fold cross-validation: Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, Extra Trees, SVR, Gradient Boosting, AdaBoost, XGBoost and LightGBM.
Then I tuned the top 3 with GridSearchCV.

🏆 Best result
SVR (C=3) – R² = 0.67 on the test set (log price), with a median error of about 25%.
The most important features were Compound, Area and City → location and size drive the price.

🚀 Deployment
I deployed the model as a Gradio web app: enter the property details and get an instant price estimate.

🔗 Live app: ADD_HUGGING_FACE_LINK
💻 GitHub: ADD_GITHUB_LINK

Thanks to NTI and our instructor for the guidance! 🙏

#MachineLearning #DataScience #Python #ScikitLearn #RealEstate #Egypt #NTI #Gradio

---
Screenshots to attach:
1. results/model_comparison.png
2. results/actual_vs_predicted.png
3. results/feature_importance.png
4. A screenshot of the live app showing a prediction
