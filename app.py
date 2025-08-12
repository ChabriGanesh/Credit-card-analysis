import streamlit as st
import pandas as pd
import joblib
# ===== Load model, scaler, and feature columns =====
model = joblib.load(r"credit-card-analysis/model.pkl")
scaler = joblib.load(r"credit-card-analysis/scaler.pkl")
X_columns = joblib.load(r"credit-card-analysis/feature_columns.pkl")  # list of feature names
# ===== Page Title =====
st.title("📊 Credit Risk Prediction App")
st.markdown("Enter applicant details below to check risk probability and approval decision.")
# ===== Risk threshold slider =====
threshold = st.slider("Risk threshold (%)", min_value=0, max_value=100, value=20, step=1) / 100
# ===== Collect Applicant Data in a Form =====
applicant_data = []
st.subheader("Applicant Details")
for col in X_columns:
    if col.endswith("_F") or col.endswith("_M") or col.endswith("_Y") or col.endswith("_N"):
        # Likely a dummy variable → use selectbox
        val = st.selectbox(f"{col} (0 = No, 1 = Yes)", [0, 1], key=col)
    else:
        val = st.number_input(f"Enter value for {col}", value=0.0, key=col)
    applicant_data.append(val)
# ===== Prediction Button =====
if st.button("Predict Risk"):
    # Create DataFrame
    applicant_df = pd.DataFrame([applicant_data], columns=X_columns)
    # Scale & Predict
    applicant_scaled = scaler.transform(applicant_df)
    prob = model.predict_proba(applicant_scaled)[0][1]
    decision = "Reject (High Risk)" if prob >= threshold else "Approve (Low Risk)"
    # Display results
    st.write(f"**Probability of High Risk:** {prob:.2%}")
    st.write(f"**Decision @ {int(threshold*100)}% threshold:** {decision}")
