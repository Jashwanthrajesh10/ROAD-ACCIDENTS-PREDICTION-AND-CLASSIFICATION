import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (update path if needed)
df = pd.read_csv("review-3-traffic-accidents.ipynb")  # Ensure correct CSV file path

# Assume last column is the target, and the rest are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target/Label

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save trained model
joblib.dump(model, "litemodel.sav")
print("✅ Model trained and saved as 'litemodel.sav'")
