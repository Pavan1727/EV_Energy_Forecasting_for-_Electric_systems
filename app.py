import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------
# Load Model & Preprocessing
# -------------------------------
model = tf.keras.models.load_model("ev_transfer_model.keras")

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
vehicle_encoder = joblib.load("vehicle_encoder.pkl")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="EV Energy Forecast", page_icon="⚡")

st.title("⚡ EV Charging Energy Forecasting")
st.write("Predict Energy Consumption & Charging Duration")

# -------------------------------
# USER INPUT UI
# -------------------------------

vehicle_model = st.selectbox(
    "Vehicle Model",
    list(vehicle_encoder.classes_)
)

battery = st.number_input("Battery Capacity (kWh)", 10.0, 200.0)
charging_rate = st.number_input("Charging Rate (kW)", 1.0, 500.0)
soc_start = st.slider("SOC Start (%)", 0.0, 100.0)
soc_end = st.slider("SOC End (%)", 0.0, 100.0)
temperature = st.number_input("Temperature (°C)", -20.0, 60.0)

days = st.slider("Select Forecast Days", 1, 30, 7)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Forecast"):

    # Encode vehicle model
    vm_encoded = vehicle_encoder.transform([vehicle_model])[0]


    # Create dataframe
    user_df = pd.DataFrame({
        "Vehicle Model":[vm_encoded],
        "Battery Capacity (kWh)":[battery],
        "Charging Rate (kW)":[charging_rate],
        "State of Charge (Start %)":[soc_start],
        "State of Charge (End %)":[soc_end],
        "Temperature (°C)":[temperature]
    })

    # Scale input
    scaled = scaler_X.transform(user_df)

    # Create sequence (same as training)
    seq = np.repeat(scaled[np.newaxis,:,:], 10, axis=1)

    predictions = []
    current_seq = seq.copy()

    # -------------------------------
    # Recursive Forecast
    # -------------------------------
    for day in range(1, days+1):

        pred_scaled = model.predict(current_seq, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)

        energy = pred[0][0]
        duration = pred[0][1]

        predictions.append([day, energy, duration])

        # Update sequence
        next_input = current_seq[0, -1, :].copy()
        next_input = next_input * (1 + 0.01*np.random.randn(len(next_input)))

        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = next_input

    # -------------------------------
    # Results Table
    # -------------------------------
    forecast_df = pd.DataFrame(
        predictions,
        columns=["Day","Energy (kWh)","Duration (hours)"]
    )

    st.subheader("📊 Prediction Results")
    st.dataframe(forecast_df)

    st.metric("Total Energy (kWh)",
              round(forecast_df["Energy (kWh)"].sum(),2))

    st.metric("Avg Charging Duration (hrs)",
              round(forecast_df["Duration (hours)"].mean(),2))

    # -------------------------------
    # TREND GRAPHS
    # -------------------------------

    st.subheader("📈 Energy Trend")

    fig1, ax1 = plt.subplots()
    ax1.plot(forecast_df["Day"], forecast_df["Energy (kWh)"])
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Energy (kWh)")
    st.pyplot(fig1)

    st.subheader("📈 Charging Duration Trend")

    fig2, ax2 = plt.subplots()
    ax2.plot(forecast_df["Day"], forecast_df["Duration (hours)"])
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Duration (hours)")
    st.pyplot(fig2)

    # Moving Average
    forecast_df["Energy_MA"] = forecast_df["Energy (kWh)"].rolling(3).mean()

    fig3, ax3 = plt.subplots()
    ax3.plot(forecast_df["Day"], forecast_df["Energy (kWh)"])
    ax3.plot(forecast_df["Day"], forecast_df["Energy_MA"])
    ax3.legend(["Actual","Moving Avg"])
    st.pyplot(fig3)

