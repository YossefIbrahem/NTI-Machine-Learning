import os
import joblib
import pandas as pd
import gradio as gr

# load the trained pipeline (imputer + scaler + Logistic Regression) saved by the notebook
folder = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(folder, 'models', 'diabetes_model.pkl'))


def predict_diabetes(hba1c, blood_sugar):
    if hba1c is None or blood_sugar is None:
        raise gr.Error('Please enter both HbA1c and blood sugar.')

    patient = pd.DataFrame([{'Blood_Sugar_Level': blood_sugar, 'HbA1c': hba1c}])
    # same feature engineering as the notebook
    patient['High_HbA1c'] = float(hba1c > 6.0)
    patient['High_Sugar'] = float(blood_sugar > 140)

    p_diabetic = float(model.predict_proba(patient)[0, 1])
    label = 'Diabetic' if p_diabetic >= 0.5 else 'Non-Diabetic'
    return label, {'Diabetic': p_diabetic, 'Non-Diabetic': 1 - p_diabetic}


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(value=5.5, label='HbA1c (%)', minimum=3, maximum=15),
        gr.Number(value=120, label='Blood Sugar Level (mg/dL)', minimum=50, maximum=400),
    ],
    outputs=[
        gr.Textbox(label='Prediction'),
        gr.Label(label='Probability'),
    ],
    examples=[
        [5.4, 120],
        [5.5, 180],
        [7.2, 130],
        [8.5, 210],
    ],
    title='Diabetes Prediction',
    description=('Screening aid trained on 9,929 patients. Model: Logistic Regression in a scikit-learn Pipeline '
                 '(test accuracy 99.75 %, F1 macro 0.992). Feature selection showed that only HbA1c and blood sugar matter. '
                 'For learning purposes only – not a medical diagnosis.'),
)

if __name__ == '__main__':
    demo.launch()
