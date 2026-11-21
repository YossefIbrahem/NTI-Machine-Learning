🏠 How much is a house in Egypt worth? I built a Machine Learning model to answer that.

I just finished the Machine Learning for Data Analysis track at NTI (National Telecommunication Institute). For my final project I predicted property prices in Egypt from 27,000+ real listings.

📊 The data
Type, area, bedrooms, bathrooms, compound, city, finishing, payment option and delivery date. Every column was stored as text, with lots of "Unknown" values.

🧹 Cleaning & preprocessing
• Converted text to numbers, merged duplicate labels and removed 1,862 duplicate rows
• Found prices that were really down payments (a 120 m² chalet for 100,000 EGP!) and removed outliers with the IQR rule on price per m², using limits learned from the training data only
• Predicted log(Price) because the price was very skewed (skewness 6.8)
• Built a Pipeline + ColumnTransformer (imputation, scaling, one-hot encoding) to avoid data leakage

🤖 Models
I compared 13 regression models with 5-fold cross-validation: Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, Extra Trees, SVR, Gradient Boosting, AdaBoost, XGBoost and LightGBM.
Then I tuned the top 3 with GridSearchCV.

🏆 Best result
SVR (C=3): R² = 0.67 on the test set (log price), with a median error of about 25%.
The most important features were Compound, Area and City, so location and size drive the price.

🚀 Deployment
I deployed the model as a web app on Hugging Face: enter the property details and get an instant price estimate. The trained scikit-learn model runs right in the browser and gives exactly the same predictions as in Python.

📚 Also in the repo
• 5 tasks, each following the same workflow (cleaning → EDA → Pipeline → model comparison → GridSearchCV → evaluation → Gradio app):
  Stroke prediction, House rent, Student performance, Customer segmentation and Diabetes prediction (99.75% accuracy)
  The Student performance and Diabetes models are live web apps too (links in the GitHub README)
• 20 practice notebooks from DEPI / Udemy: Adult Census Income, Bank term deposits, Ames Housing, Laptop prices, Used cars, Telecom churn and more

🔗 Live app: https://huggingface.co/spaces/YoussefIbrahem0/egypt-house-price-prediction
💻 GitHub: https://github.com/YossefIbrahem/NTI-Machine-Learning

Thanks to NTI and our instructor for the guidance! 🙏

#MachineLearning #DataScience #Python #ScikitLearn #RealEstate #Egypt #NTI #DEPI #HuggingFace

---
Screenshots to attach:
1. results/model_comparison.png
2. results/actual_vs_predicted.png
3. results/feature_importance.png
4. A screenshot of the live app showing a prediction
