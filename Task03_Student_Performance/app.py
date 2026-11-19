import os
import joblib
import pandas as pd
import gradio as gr

# load the trained pipeline (imputers + scaler + encoder + Ridge model) saved by the notebook
folder = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(folder, 'models', 'student_model.pkl'))


def predict_performance(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers):
    student = pd.DataFrame([{
        'Hours Studied': hours_studied,
        'Previous Scores': previous_scores,
        'Sleep Hours': sleep_hours,
        'Sample Question Papers Practiced': sample_papers,
        'Extracurricular Activities': extracurricular,
    }])
    score = float(model.predict(student)[0])
    score = min(max(score, 10), 100)   # the index is between 10 and 100

    if score >= 70:
        level = 'High performance'
    elif score >= 40:
        level = 'Average performance'
    else:
        level = 'Low performance – needs support'
    return round(score, 1), level


demo = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Slider(1, 9, value=5, step=1, label='Hours Studied (per day)'),
        gr.Slider(40, 99, value=70, step=1, label='Previous Scores'),
        gr.Radio(['Yes', 'No'], value='No', label='Extracurricular Activities'),
        gr.Slider(4, 9, value=7, step=1, label='Sleep Hours'),
        gr.Slider(0, 9, value=5, step=1, label='Sample Question Papers Practiced'),
    ],
    outputs=[
        gr.Number(label='Predicted Performance Index (10 – 100)'),
        gr.Textbox(label='Level'),
    ],
    examples=[
        [7, 85, 'Yes', 8, 5],
        [2, 45, 'No', 5, 1],
        [5, 70, 'No', 7, 3],
    ],
    title='Student Performance Prediction',
    description=('Predict a student\'s Performance Index from study habits and previous scores. '
                 'Model: Ridge regression in a scikit-learn Pipeline (test R² = 0.989, average error ≈ 1.7 points).'),
)

if __name__ == '__main__':
    demo.launch()
