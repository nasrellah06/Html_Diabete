from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("RForest_model.pkl")

features = ['Age','Pregnancies','BMI','Glucose','BloodPressure','HbA1c','LDL','HDL','Triglycerides','FamilyHistory','DietType','Hypertension','MedicationUse']

def preprocess(form):
    data = {}
    for key in features:
        value = form.get(key)
        if value is None or value == "":
            raise ValueError(f"Le champ '{key}' est manquant ou vide.")
        data[key] = value

    df = pd.DataFrame([data])

    # Convertir les colonnes numériques
    numeric_cols = ['Age','Pregnancies','BMI','Glucose','BloodPressure','HbA1c','LDL','HDL','Triglycerides','FamilyHistory','DietType','Hypertension','MedicationUse']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='raise')

    
    df_encoded = pd.get_dummies(df)

    # Aligner avec les colonnes du modèle
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_encoded:
            df_encoded[col] = 0

    return df_encoded[model_columns]


@app.route('/')

def home():
    return render_template('index_diabete.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = preprocess(request.form)
    prediction = model.predict(input_data)[0]
    result = "Le client est malade" if prediction == 1 else "Le client est saint"
    return render_template('index_diabete.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)