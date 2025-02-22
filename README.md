# Heart Attack Risk Prediction App

This is a **Heart Attack Risk Prediction App** built using Python, Flask, and Scikit-learn. The app predicts the risk of a heart attack based on user-provided health metrics such as age, cholesterol, blood pressure, BMI, and more.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Structure](#code-structure)
6. [Example Inputs](#example-inputs)
7. [License](#license)

---

## Overview

The app uses a **Random Forest Classifier** trained on a dataset containing health and lifestyle factors to predict the risk of a heart attack. The model is deployed as a web application using **Flask**, allowing users to input their health metrics and receive a prediction.

---

## Features

- **Input Form**: Users can input health metrics such as age, gender, cholesterol, blood pressure, BMI, and more.
- **Prediction**: The app predicts whether the user is at **High Risk** or **Low Risk** of a heart attack.
- **Scalable**: The app can be deployed locally or on a cloud platform like Heroku, AWS, or Google Cloud.

---

## Installation

### Prerequisites
- Python 3.x
- Pip (Python package manager)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/heart-attack-risk-prediction-app.git
   cd heart-attack-risk-prediction-app
Create a Virtual Environment:

bash
Copy
python -m venv venv
Activate the Virtual Environment:

On Windows:

bash
Copy
venv\Scripts\activate
On macOS/Linux:

bash
Copy
source venv/bin/activate
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Download the Dataset:

Place the dataset (heart_attack_data.csv) in the project folder.

Usage
1. Train the Model
Run the train_model.py script to preprocess the data, train the model, and save the model, feature names, and scaler:

bash
Copy
python train_model.py
2. Run the Flask App
Start the Flask app:

bash
Copy
python app.py
3. Access the App
Open your browser and go to http://127.0.0.1:5000/. Enter the required health metrics and click Predict Risk.

Code Structure
Copy
heart-attack-risk-prediction-app/
│
├── app.py                  # Flask app for prediction
├── train_model.py          # Script to train and save the model
├── templates/
│   └── index.html          # HTML template for the input form
├── static/
│   └── style.css           # CSS file for styling the app
├── heart_attack_model.pkl  # Trained model
├── feature_names.pkl       # Saved feature names
├── scaler.pkl              # Saved scaler for feature scaling
└── heart_attack_data.csv   # Dataset for training
Example Inputs
Example 1: Low Risk of Heart Attack
Age: 35

Gender: Female

Cholesterol: 150 mg/dL

Blood Pressure: 120 mmHg

Heart Rate: 70 bpm

BMI: 22.5

Smoker: 0 (No)

Diabetes: 0 (No)

Hypertension: 0 (No)

Family History: 0 (No)

Expected Prediction: Low Risk of Heart Attack

Example 2: High Risk of Heart Attack
Age: 65

Gender: Male

Cholesterol: 280 mg/dL

Blood Pressure: 160 mmHg

Heart Rate: 95 bpm

BMI: 35.0

Smoker: 1 (Yes)

Diabetes: 1 (Yes)

Hypertension: 1 (Yes)

Family History: 1 (Yes)

Expected Prediction: High Risk of Heart Attack

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Dataset: Heart Attack Prediction Dataset

Libraries: Flask, Scikit-learn, Pandas, Joblib

Copy

---

### **How to Use the `README.md`**

1. Save the above content in a file named `README.md` in your project folder.
2. Update the placeholders (e.g., `your-username`, `your-dataset-link`) with your actual information.
3. Commit the `README.md` file to your repository:
   ```bash
   git add README.md
   git commit -m "Add README.md file"
   git push origin main
This README.md file provides a clear and concise guide for anyone who wants to use or contribute to your project. Let me know if you need further assistance! 🚀
