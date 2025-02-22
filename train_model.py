# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

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

# Step 7: Handle Imbalanced Data
print("\nHandling imbalanced data...")

# Separate majority and minority classes
df_majority = df[df['Outcome'] == 'No Heart Attack']
df_minority = df[df['Outcome'] == 'Heart Attack']

# Upsample the minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # Sample with replacement
                                 n_samples=len(df_majority),  # Match majority class size
                                 random_state=42)

# Combine the majority class with the upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Check the new class distribution
print(df_upsampled['Outcome'].value_counts())

# Update X_train and y_train with the upsampled data
X_train = df_upsampled.drop('Outcome', axis=1)
y_train = df_upsampled['Outcome']

# Step 8: Scale Numerical Features
print("\nScaling numerical features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Train a Machine Learning Model
print("\nTraining the Random Forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# Save the model to a file
joblib.dump(model, 'heart_attack_model.pkl')
print("Model saved as 'heart_attack_model.pkl'.")

# Step 10: Evaluate the Model
print("\nEvaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')