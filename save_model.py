
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the existing dataset
data = pd.read_csv("dataset/energydata_complete.csv")

# Prepare date features
data["date"] = pd.to_datetime(data["date"])
data["hour"] = data["date"].dt.hour
data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month

# Separate input features and target
X = data.drop(columns=["Appliances", "date"])
y = data["Appliances"]

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Store typical values for features not entered on the website
defaults = X.median(numeric_only=True).to_dict()

# Save model with compression to reduce file size
model_data = {
    "model": model,
    "features": list(X.columns),
    "defaults": defaults
}

joblib.dump(
    model_data,
    "energy_model.pkl",
    compress=9
)

print("Random Forest model saved successfully!")
print("Total input features:", len(X.columns))
print("Saved file: energy_model.pkl")

# Display the saved model file size
import os

file_size_mb = os.path.getsize("energy_model.pkl") / (1024 * 1024)
print(f"Compressed model size: {file_size_mb:.2f} MB")
