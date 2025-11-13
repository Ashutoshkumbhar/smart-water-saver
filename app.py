import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------
# 🔹 Load Trained Model and Scaler
# -------------------------------
model = joblib.load("models/pune_water_zone_model.pkl")
scaler = joblib.load("models/pune_scaler.pkl")

# -------------------------------
# 🔹 Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Smart Water Saver", layout="centered")
st.title("💧 Smart Water Saver – Pune Zone")
st.markdown("Real-time IoT-based **Water Consumption Prediction** and **Visualization**")

# Sidebar inputs for IoT parameters
st.sidebar.header("📡 IoT Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 25)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 60)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 10.0)
population = st.sidebar.number_input("Population (in 1000s)", 0, 10000, 200)
day = st.sidebar.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

# Convert day to numerical feature
day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
day_val = day_map[day]

# -------------------------------
# 🔹 Single Prediction Button
# -------------------------------
st.subheader("🔍 Predict Water Usage")
if st.button("Predict Now"):
    input_data = np.array([[temperature, humidity, rainfall, population, day_val]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    st.success(f"💧 Predicted Water Usage: **{prediction:.2f} litres**")

# -------------------------------
# 🔹 Real-Time Simulation Section
# -------------------------------
st.subheader("📈 Live IoT Simulation")

placeholder = st.empty()
usage_values = []
timestamps = []

simulate = st.checkbox("Start Real-Time Simulation")

if simulate:
    for i in range(15):  # simulate for 15 seconds
        # Simulated live sensor values
        temp = np.random.uniform(temperature - 2, temperature + 2)
        hum = np.random.uniform(humidity - 5, humidity + 5)
        rain = np.random.uniform(max(0, rainfall - 5), rainfall + 5)
        pop = population
        d = day_val

        # Scale and predict
        features = np.array([[temp, hum, rain, pop, d]])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]

        usage_values.append(pred)
        timestamps.append(datetime.now().strftime("%H:%M:%S"))

        # Live chart
        fig, ax = plt.subplots()
        ax.plot(timestamps, usage_values, marker='o')
        ax.set_title("Live Predicted Water Usage")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Predicted Usage (Litres)")
        plt.xticks(rotation=45)
        placeholder.pyplot(fig)

        time.sleep(1)

    st.success("✅ Simulation Completed!")

# -------------------------------
# 🔹 Footer
# -------------------------------
st.markdown("---")
st.caption("Developed for Pune Smart Water Saver Project 💧 | Powered by Streamlit & Machine Learning")
