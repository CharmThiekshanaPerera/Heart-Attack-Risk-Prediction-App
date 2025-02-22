from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_attack_model.pkl')

# Load the feature names
feature_names = joblib.load('feature_names.pkl')

# Load the scaler
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    cholesterol = float(request.form['cholesterol'])
    blood_pressure = float(request.form['blood_pressure'])
    heart_rate = float(request.form['heart_rate'])
    bmi = float(request.form['bmi'])
    smoker = int(request.form['smoker'])
    diabetes = int(request.form['diabetes'])
    hypertension = int(request.form['hypertension'])
    family_history = int(request.form['family_history'])

    # Convert gender to numerical
    gender = 1 if gender == 'Male' else 0

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Cholesterol': [cholesterol],
        'BloodPressure': [blood_pressure],
        'HeartRate': [heart_rate],
        'BMI': [bmi],
        'Smoker': [smoker],
        'Diabetes': [diabetes],
        'Hypertension': [hypertension],
        'FamilyHistory': [family_history]
    })

    # Add missing columns with default values (0)
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match the training data
    input_data = input_data[feature_names]

    # Scale the input data
    input_data_scaled = scaler.fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    result = 'High Risk of Heart Attack' if prediction == 1 else 'Low Risk of Heart Attack'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)