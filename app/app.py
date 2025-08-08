# Heart Disease Risk Prediction App
# ----------------------------------------------------------------
# This Streamlit app loads a trained XGBoost model and allows users
# to input patient details to predict the risk of heart disease.
# It also provides SHAP-based explanations for model predictions.

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Heart Disease Risk Prediction")
st.markdown(
    """
    This app predicts the **probability of heart disease** based on patient health parameters.
    It uses a trained **XGBoost model** and explains predictions using SHAP values.
    """
)

# --- Load Model, Scaler, and Feature Names ---
try:
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl") 
except FileNotFoundError:
    st.error("Model, scaler, or feature_names file not found. Please make sure they are in the app directory.")
    st.stop()

# --- Sidebar Input Form ---
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 500, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
thalch = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

# --- Encode categorical inputs ---
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# --- Build input dictionary with only base features ---
base_input = {
    "age": age,
    "sex": sex,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "ca": ca
}

# --- Create full feature DataFrame in correct order ---
input_df = pd.DataFrame([base_input])

# Add missing one-hot encoded columns as 0
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns exactly as training
input_df = input_df[feature_names]

# --- Scale ---
input_scaled = scaler.transform(input_df)

# --- Prediction ---
if st.sidebar.button("Predict Risk"):
    pred_class = model.predict(input_scaled)[0]
    pred_prob = model.predict_proba(input_scaled)[0][1]

    if pred_class == 1:
        st.error(f"'High risk' of heart disease detected! (Probability: {pred_prob:.2%})")
    else:
        st.success(f"'Low risk' of heart disease. (Probability: {pred_prob:.2%})")

    # --- SHAP Explanations ---
    st.subheader("Model Feature Importance")
    explainer = shap.Explainer(model, pd.DataFrame(input_scaled, columns=feature_names))
    shap_values = explainer(pd.DataFrame(input_scaled, columns=feature_names))

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, pd.DataFrame(input_scaled, columns=feature_names), plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("Explanation for This Prediction")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap_values[0], show=False)
    st.pyplot(fig2)

