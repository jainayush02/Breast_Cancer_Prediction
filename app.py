# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --- 1. LOAD THE TRAINED MODEL AND SCALER ---
model = None
scaler = None

# Load the trained neural network model
try:
    # Use compile=False to avoid optimizer/loss issues when loading
    model = tf.keras.models.load_model("model.keras", compile=False)
except Exception as e:
    st.error(f"❌ Error loading the model (model.keras): {e}")
    st.stop()

# Load the scaler object using joblib
try:
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"❌ Error loading the scaler file (scaler.joblib): {e}")
    st.stop()


# --- 2. DEFINE THE USER INTERFACE ---
st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", layout="wide")
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.write("""
This application uses an Artificial Neural Network to predict whether a breast tumor is **Malignant** or **Benign**.
Please adjust the values below to input the patient's diagnostic measurements.
""")
st.markdown("---")


# Feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Layout for input
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]
user_input = {}

st.subheader("Patient's Measurements")

# Input fields
for i, feature in enumerate(feature_names):
    with columns[i % 3]:
        user_input[feature] = st.number_input(
            label=f'{feature.replace("_", " ").title()}',
            min_value=0.0,
            max_value=5000.0,
            value=10.0,
            step=0.1,
            key=feature
        )

# --- 3. PROCESS INPUT AND MAKE PREDICTION ---
st.markdown("---")
if st.button('**Predict Diagnosis**', use_container_width=True):
    if model is not None and scaler is not None:
        # Convert input to DataFrame and order correctly
        input_df = pd.DataFrame([user_input])[feature_names]
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction_prob = model.predict(input_scaled)
        prediction = int((prediction_prob > 0.5).astype(int).flatten()[0])

        # --- 4. DISPLAY THE RESULT ---
        st.header("Prediction Result")
        if prediction == 1:
            st.error("Diagnosis: **Malignant**", icon="🚨")
            st.write("The model predicts that the tumor is cancerous. Please consult with a medical professional for further evaluation and confirmation.")
        else:
            st.success("Diagnosis: **Benign**", icon="✅")
            st.write("The model predicts that the tumor is non-cancerous. However, it is always recommended to follow up with a healthcare provider.")

        # Confidence score
        confidence = prediction_prob[0][0] if prediction == 1 else 1 - prediction_prob[0][0]
        st.subheader("Model Confidence")
        st.info(f"The model is **{confidence*100:.2f}%** confident in its prediction.")
    else:
        st.warning("⚠️ Model or scaler not loaded. Please check the terminal logs.")


st.markdown("---")
st.write("Disclaimer: This is a machine learning application and not a substitute for professional medical advice.")
