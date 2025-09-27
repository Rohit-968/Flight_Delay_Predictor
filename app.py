import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="✈️ Flight Delay Predictor", layout="centered")
st.title("✈️ Flight Delay Predictor")

# Features used in training
MODEL_FEATURES = ["DEPARTURE_DELAY", "DISTANCE", "AIR_TIME", "MONTH"]

@st.cache_data
def load_data(sample_rows=None):
    
    df = pd.read_csv("flights.csv", usecols=MODEL_FEATURES)
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=42)
    return df

df = load_data(sample_rows=10000)  # 10k rows to start


st.subheader("Dataset Preview (first 100 rows only)")
st.dataframe(df.head(100))
st.write(f"Total rows: {df.shape[0]}, columns: {df.shape[1]}")

# Load model
model = None
try:
    model = joblib.load("flight_delay_model.pkl")
    st.success("✅ Loaded flight_delay_model.pkl")
except Exception as e:
    st.warning(f"Model not found or failed to load: {e}")

st.subheader("Single Flight Prediction")

if model is not None:
    st.info(f"Enter values for features used in the model: {MODEL_FEATURES}")
    
    departure_delay = st.number_input("Departure Delay (minutes)", value=0)
    distance = st.number_input("Distance (miles)", value=500)
    air_time = st.number_input("Air Time (minutes)", value=100)
    month = st.slider("Month", 1, 12, 7)
    
    input_df = pd.DataFrame({
        "DEPARTURE_DELAY": [departure_delay],
        "DISTANCE": [distance],
        "AIR_TIME": [air_time],
        "MONTH": [month]
    })
    
    st.write("Input Preview:")
    st.table(input_df)
    
    if st.button("Predict This Flight"):
        try:
            prediction = model.predict(input_df)
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
            
            if prediction[0] == 1:
                msg = f"🚨 Flight is likely DELAYED"
            else:
                msg = f"✅ Flight is likely ON TIME"
            
            if prob is not None:
                msg += f" (Probability: {prob:.2f})"
            
            st.write(msg)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("No model loaded — single-flight prediction unavailable.")

st.subheader("Full-Dataset Batch Prediction")

if model is not None:
    if st.button("Run Prediction on Full Dataset"):
        try:
            df_copy = df.copy()
            
            # Select only features used by the model
            df_for_pred = df_copy[MODEL_FEATURES]
            
            df_copy["prediction"] = model.predict(df_for_pred)
            df_copy.to_csv("predicted_output.csv", index=False)
            
            st.success("✅ Predictions done! Saved as predicted_output.csv")
            st.download_button("Download Predictions CSV", "predicted_output.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("No model loaded — cannot run batch predictions.")
