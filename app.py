# Heart Disease Risk Prediction App
# ----------------------------------------------------------------
# Streamlit app to predict heart disease risk using a trained XGBoost model.
# Allows patient data input, predicts probability, and explains results with SHAP.

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# ---------------- App Config ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Heart Disease Risk Prediction")
st.markdown("""
    This app predicts the **probability of heart disease** from patient health data.
    It uses a trained **XGBoost** model and SHAP values for explanation.
""")

# ---------------- Load Model & Data ----------------
try:
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    # Load test set for global feature importance plot
    X_test_scaled = joblib.load("X_test_scaled.pkl")
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 500, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
thalch = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

# ---------------- Encode Inputs ----------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

patient_data = {
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

# Make sure all expected features are present
input_df = pd.DataFrame([patient_data])
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Scale input
input_scaled = scaler.transform(input_df)

# ---------------- Prediction ----------------
if st.sidebar.button("Predict Risk"):
    pred_class = model.predict(input_scaled)[0]
    pred_prob = model.predict_proba(input_scaled)[0][1]

    if pred_class == 1:
        st.error(f"High risk of heart disease detected! (Probability: {pred_prob:.2%})")
    else:
        st.success(f"Low risk of heart disease. (Probability: {pred_prob:.2%})")

    # ----- Global Feature Importance -----
    st.subheader("Model Feature Importance")
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    explainer_global = shap.Explainer(model, X_test_df)
    shap_values_global = explainer_global(X_test_df)

    shap.summary_plot(shap_values_global, X_test_df, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # ----- Local Prediction Explanation -----
    st.subheader("Explanation for This Prediction")
    explainer_local = shap.Explainer(model, pd.DataFrame(input_scaled, columns=feature_names))
    shap_values_local = explainer_local(pd.DataFrame(input_scaled, columns=feature_names))

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap_values_local[0], show=False)
    st.pyplot(fig)
