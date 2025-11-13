import pandas as pd
import numpy as np
import joblib
import time
import random
from datetime import datetime

# -------------------------------
# 1️⃣ Load model and scaler
# -------------------------------
model = joblib.load("models/pune_water_zone_model.pkl")
scaler = joblib.load("models/pune_scaler.pkl")

print("✅ Smart Water Saver — IoT Simulation Started!\n")

# -------------------------------
# 2️⃣ Simulate sensor data
# -------------------------------
def generate_sensor_data():
    """Simulates live readings from water flow sensors in each zone."""
    return {
        "Parvati M.L.D": random.uniform(400, 460),
        "New & Old cantonment M.L.D": random.uniform(10, 20),
        "Waraje Close Pipe M.L.D": random.uniform(15, 25),
        "Old Holkar M.L.D": random.uniform(9, 13),
        "Vadgaon Close Pipe M.L.D": random.uniform(150, 180),
        "Day": datetime.now().day,
        "Month": datetime.now().month,
        "Weekday": datetime.now().weekday()
    }

# -------------------------------
# 3️⃣ Live prediction loop
# -------------------------------
try:
    while True:
        # Get live data
        sensor_data = generate_sensor_data()
        df_live = pd.DataFrame([sensor_data])

        # Match training columns
        try:
            expected_features = model.feature_names_in_
        except AttributeError:
            expected_features = df_live.columns.tolist()
        df_live = df_live.reindex(columns=expected_features, fill_value=0)

        # Scale + predict
        scaled = scaler.transform(df_live)
        predicted_total = model.predict(scaled)[0]

        # Print results
        print(f"\n🕒 {datetime.now().strftime('%H:%M:%S')}")
        print(f"💧 Predicted Total Water Supply (MLD): {predicted_total:.2f}")

        # Alert logic
        if predicted_total > 1200:
            print("⚠️ ALERT: High water usage detected! Reduce irrigation or check for leaks.")
        elif predicted_total < 800:
            print("✅ Optimal usage — systems running efficiently.")
        else:
            print("ℹ️ Moderate usage — normal operation.")

        time.sleep(5)  # wait 5 seconds before next reading

except KeyboardInterrupt:
    print("\n🛑 Simulation stopped by user.")
