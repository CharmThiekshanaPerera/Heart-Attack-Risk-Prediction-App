# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score

# Step 1: Load the Dataset
df = pd.read_csv('heart_attack_data.csv')
print("Dataset loaded successfully!")

# Step 2: Check Column Names
print("\nColumn names in the dataset:")
print(df.columns)

# Step 3: Handle Missing Values
print("\nHandling missing values...")
print("Missing values before handling:")
print(df.isnull().sum())

# Fill missing values for numeric columns only
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

print("\nMissing values after handling:")
print(df.isnull().sum())

# Step 4: Convert Categorical Variables
print("\nConverting categorical variables...")

# Encode categorical columns
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['ChestPainType'] = df['ChestPainType'].map({'Typical': 0, 'Atypical': 1, 'Non-anginal': 2, 'Asymptomatic': 3})
df['ExerciseInducedAngina'] = df['ExerciseInducedAngina'].map({'Yes': 1, 'No': 0})
df['Slope'] = df['Slope'].map({'Upsloping': 0, 'Flat': 1, 'Downsloping': 2})
df['Thalassemia'] = df['Thalassemia'].map({'Normal': 0, 'Fixed defect': 1, 'Reversible defect': 2})
df['StressLevel'] = df['StressLevel'].map({'Low': 0, 'Moderate': 1, 'High': 2})

# Save the target column before encoding
target = df['Outcome']

# Use pd.get_dummies for categorical columns (excluding 'Outcome')
df = pd.get_dummies(df.drop('Outcome', axis=1), drop_first=True)

# Add the target column back to the DataFrame
df['Outcome'] = target

print("\nCategorical variables converted to numeric.")

# Step 5: Verify the Target Column
if 'Outcome' not in df.columns:
    raise ValueError("Target column 'Outcome' not found in the dataset after preprocessing.")

# Step 6: Split the Dataset
print("\nSplitting the dataset into features (X) and target (y)...")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")

# Step 7: Train a Machine Learning Model
print("\nTraining the Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Save the model to a file
joblib.dump(model, 'heart_attack_model.pkl')
print("Model saved as 'heart_attack_model.pkl'.")

# Step 8: Evaluate the Model
print("\nEvaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')