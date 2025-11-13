import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime

# ✅ Load trained model and scaler
model = joblib.load(r"A:\smart_water_saver\models\pune_water_zone_model.pkl")
scaler = joblib.load(r"A:\smart_water_saver\models\pune_scaler.pkl")

print("✅ Both files loaded successfully!")

# ✅ Prepare live plot
plt.ion()
fig, ax = plt.subplots()
times, predictions = [], []

# ✅ Start timer
start_time = time.time()

# ✅ Prepare CSV log
log_file = r"A:\smart_water_saver\live_predictions.csv"
log_data = []

while True:
    # Stop after 5 seconds
    if time.time() - start_time > 10:
        print("⏹️ Visualization stopped after 5 seconds.")
        break

    # 🔹 Generate live sensor-like input (8 features)
    sample = np.array([[np.random.uniform(300, 500),  # Parvati MLD
                        np.random.uniform(50, 150),   # New & Old Cantonment MLD
                        np.random.uniform(100, 250),  # Waraje Close Pipe MLD
                        np.random.uniform(10, 15),    # Old Holkar MLD
                        np.random.uniform(100, 200),  # Vadgaon Close Pipe MLD
                        datetime.now().day,
                        datetime.now().month,
                        datetime.now().weekday()]])
    
    # 🔹 Scale and predict
    scaled = scaler.transform(sample)
    prediction = model.predict(scaled)[0]

    # 🔹 Log timestamp and prediction
    timestamp = datetime.now().strftime("%H:%M:%S")
    times.append(timestamp)
    predictions.append(prediction)
    log_data.append([timestamp, *sample[0], prediction])

    # 🔹 Update live plot
    ax.clear()
    ax.plot(times, predictions, marker='o', color='blue')
    ax.set_xlabel("Time")
    ax.set_ylabel("Predicted Water Usage (MLD)")
    ax.set_title("Live Water Consumption Prediction - Pune")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.pause(1)  # refresh every second

# 🔹 Save predictions log
df_log = pd.DataFrame(log_data, columns=[
    "Time", "Parvati MLD", "New&Old Cantonment MLD",
    "Waraje Close Pipe MLD", "Old Holkar MLD", "Vadgaon Close Pipe MLD",
    "Day", "Month", "Weekday", "Predicted Total MLD"
])
df_log.to_csv(log_file, index=False)
print(f"💾 Predictions saved to {log_file}")

plt.ioff()
plt.show()
