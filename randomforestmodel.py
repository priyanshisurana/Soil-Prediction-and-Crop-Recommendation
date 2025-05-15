 #dataset1-crop recommendation
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
 # Load the dataset
file_path = r"C:\Users\vagis\OneDrive\Desktop\indiancrop_dataset.csv"
data = pd.read_csv(file_path)
 # Data Cleaning
 # Check for missing values
if data.isnull().sum().sum() > 0:
    print("Handling missing values...")
    data.fillna(data.mean(), inplace=True)
 # Check for duplicates and remove them
if data.duplicated().sum() > 0:
    print("Removing duplicates...")
    data = data.drop_duplicates()
 # Verify data ranges (example: pH should be between 4.5 and 8.5)
data = data[(data['ph'] >= 4.5) & (data['ph'] <= 8.5)]
 # Exploratory Data Analysis (Optional)
print("Data overview:")
print(data.describe())
print("\nClass Distribution:")
print(data['CROP'].value_counts())
  # Define features (X) and target (y)
X = data.drop(columns=['CROP'])  # Features
y = data['CROP']  # Target
 
# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
 
# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
 
# Accept user input for crop recommendation
print("\nEnter the following details for crop recommendation:")
N = float(input("Enter Nitrogen content in soil (N): "))
P = float(input("Enter Phosphorous content in soil (P): "))
K = float(input("Enter Potassium content in soil (K): "))
temperature = float(input("Enter temperature (°C): "))
humidity = float(input("Enter humidity (%): "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter rainfall (mm): "))
 
# Prepare input for prediction
user_input = pd.DataFrame({
    'N': [N],
    'P': [P],
    'K': [K],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})
 
# Match the format of the dataset
user_input_scaled = scaler.transform(user_input)
 
# Predict the crop
predicted_crop = rf_model.predict(user_input_scaled)
print(f"\nRecommended Crop: {predicted_crop[0]}")
import pickle

# Save the trained Random Forest model as .pkl
with open('rf_soil_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)  # Save the Random Forest model as .pkl
