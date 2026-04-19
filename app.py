import streamlit as st
import pandas as pd
import joblib


# Title and description
st.title("🧬 Embryo Quality Prediction ")
st.markdown("Welcome ** Embryo Quality Predictor using clinical parameters.")
st.markdown("---")

# Load model and encoder
model = joblib.load(r"C:\embro_quality\xgb_embryo_model.pkl")
label_encoder = joblib.load(r"C:\embro_quality\label_encoder.pkl")

# Input fields
st.header("📋 Enter Patient Clinical Parameters")

FSH = st.number_input("FSH (mIU/mL)", min_value=0.0, step=0.1)
LH = st.number_input("LH (mIU/mL)", min_value=0.0, step=0.1)
Age = st.number_input("Age (yrs)", min_value=0, step=1)
AMH = st.number_input("AMH (ng/mL)", min_value=0.0, step=0.1)
BMI = st.number_input("BMI", min_value=0.0, step=0.1)
AFC = st.number_input("AFC", min_value=0, step=1)

# Prediction
if st.button("🔍 Predict Embryo Quality"):
    input_df = pd.DataFrame([[FSH, LH, Age, AMH, BMI, AFC]],
                            columns=['FSH(mIU/mL)', 'LH(mIU/mL)', 'Age (yrs)', 'AMH(ng/mL)', 'BMI', 'AFC'])

    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 Predicted Embryo Quality: **{predicted_label}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)
        prob_df = pd.DataFrame(probs, columns=label_encoder.classes_)
        st.subheader("📊 Prediction Probabilities")
        st.dataframe(prob_df.T.style.format("{:.2%}"))
