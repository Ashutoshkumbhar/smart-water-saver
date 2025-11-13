import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------
# 1️⃣ Load trained model and scaler
# -------------------------------
model = joblib.load("models/pune_water_zone_model.pkl")
scaler = joblib.load("models/pune_scaler.pkl")

print("✅ Model and Scaler Loaded Successfully!")

# -------------------------------
# 2️⃣ Define input data (replace with live/sensor inputs)
# -------------------------------
# Example: Suppose today's readings are as follows 👇
new_data = {
    "Parvati M.L.D": 420.5,
    "New & Old cantonment M.L.D": 15.2,
    "Waraje Close Pipe M.L.D": 22.4,
    "Old Holkar M.L.D": 10.9,
    "Vadgaon Close Pipe M.L.D": 165.3,
    "Day": datetime.now().day,
    "Month": datetime.now().month,
    "Weekday": datetime.now().weekday()
}

# Convert to DataFrame
df_new = pd.DataFrame([new_data])

# -------------------------------
# 3️⃣ Determine expected features
# -------------------------------
try:
    expected_features = model.feature_names_in_
except AttributeError:
    print("⚙️ 'feature_names_in_' not found — inferring from scaler or columns...")
    try:
        expected_features = scaler.feature_names_in_
    except AttributeError:
        expected_features = df_new.columns.tolist()  # fallback if nothing saved

# -------------------------------
# 4️⃣ Align columns
# -------------------------------
df_new = df_new.reindex(columns=expected_features, fill_value=0)

# -------------------------------
# 5️⃣ Scale and Predict
# -------------------------------
df_scaled = scaler.transform(df_new)
predicted_total = model.predict(df_scaled)[0]

print(f"\n💧 Predicted Total Water Supply (MLD): {predicted_total:.2f}")

# -------------------------------
# 6️⃣ Decision Support
# -------------------------------
if predicted_total > 1200:
    print("⚠️ High water usage expected — consider irrigation control.")
elif predicted_total < 800:
    print("✅ Water usage within optimal range.")
else:
    print("ℹ️ Moderate usage — monitor tank levels.")
