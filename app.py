# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('heart_attack_data.csv')

# Convert categorical variables (e.g., Gender) to numerical
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'heart_attack_model.pkl')