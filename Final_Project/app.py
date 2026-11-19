import os
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# load the trained pipeline (preprocessing + model) and the dropdown values saved by the notebook
folder = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(folder, 'models', 'house_price_model.pkl'))
with open(os.path.join(folder, 'models', 'choices.json')) as f:
    choices = json.load(f)


def predict_price(Type, Bedrooms, Bathrooms, Area, Furnished, Level, Compound,
                  Payment_Option, Delivery_Date, Delivery_Term, City):
    if not Area or not Bedrooms or not Bathrooms:
        raise gr.Error('Please enter Area, Bedrooms and Bathrooms.')

    house = pd.DataFrame([{
        'Type': Type, 'Bedrooms': Bedrooms, 'Bathrooms': Bathrooms, 'Area': Area,
        'Furnished': Furnished, 'Level': Level, 'Compound': Compound,
        'Payment_Option': Payment_Option, 'Delivery_Date': Delivery_Date,
        'Delivery_Term': Delivery_Term, 'City': City,
    }])

    # same feature engineering as the notebook
    house['Log_Area'] = np.log1p(house['Area'])
    house['Area_per_Bedroom'] = house['Area'] / house['Bedrooms']

    price = np.expm1(model.predict(house)[0])   # the model predicts log(Price)
    return f'{price:,.0f} EGP', f'{price / Area:,.0f} EGP per m²'


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(choices['Type'], value='Apartment', label='Type'),
        gr.Number(value=3, label='Bedrooms', minimum=1, maximum=10, precision=0),
        gr.Number(value=2, label='Bathrooms', minimum=1, maximum=10, precision=0),
        gr.Number(value=150, label='Area (m²)', minimum=10, maximum=1000),
        gr.Dropdown(choices['Furnished'], value='No', label='Furnished'),
        gr.Dropdown(choices['Level'], value='3', label='Level (floor)'),
        gr.Dropdown(choices['Compound'], value='Unknown', label='Compound'),
        gr.Dropdown(choices['Payment_Option'], value='Cash', label='Payment Option'),
        gr.Dropdown(choices['Delivery_Date'], value='Ready to move', label='Delivery Date'),
        gr.Dropdown(choices['Delivery_Term'], value='Finished', label='Delivery Term (finishing)'),
        gr.Dropdown(choices['City'], value='New Cairo - El Tagamoa', label='City'),
    ],
    outputs=[
        gr.Textbox(label='Predicted price'),
        gr.Textbox(label='Price per m²'),
    ],
    examples=[
        ['Apartment', 3, 2, 150, 'No', '3', 'Unknown', 'Cash', 'Ready to move', 'Finished', 'New Cairo - El Tagamoa'],
        ['Chalet', 2, 2, 110, 'No', 'Ground', 'Marassi', 'Cash or Installment', '2024', 'Finished', 'North Coast'],
        ['Stand Alone Villa', 5, 5, 450, 'No', 'Unknown', 'Unknown', 'Cash', 'Ready to move', 'Semi Finished', 'Sheikh Zayed'],
    ],
    title='Egypt House Price Prediction',
    description=('Estimate the price of a property in Egypt. '
                 'Model: SVR tuned with GridSearchCV, trained on 25k listings '
                 '(test R² = 0.67 on log price, typical error ≈ 25%).'),
)

if __name__ == '__main__':
    demo.launch()
