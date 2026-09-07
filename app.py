import joblib
import pandas as pd
import streamlit as st

# IMPORTANT: the pipeline was saved with joblib and contains custom
# transformer classes (RowWiseFeatureAdder, RiskScoreAdder, OutlierCapper).
# joblib/pickle needs these exact classes importable at load time, so this
# import must happen BEFORE joblib.load, and preprocessing.py must sit next
# to this file.
from preprocessing import RowWiseFeatureAdder, RiskScoreAdder, OutlierCapper  # noqa: F401

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load("heart_disease_pipeline.joblib")


pipeline = load_pipeline()

st.title("❤️ Heart Disease Risk Predictor")
st.write("Enter the patient's clinical values below to estimate heart disease risk.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=54)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Chest pain type",
            options=[
                ("Typical angina", 0),
                ("Atypical angina", 1),
                ("Non-anginal pain", 2),
                ("Asymptomatic", 3),
            ],
            format_func=lambda x: x[0],
        )[1]
        trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=130)
        chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600, value=245)
        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0],
        )[1]
        restecg = st.selectbox(
            "Resting ECG results",
            options=[
                ("Normal", 0),
                ("ST-T wave abnormality", 1),
                ("Left ventricular hypertrophy", 2),
            ],
            format_func=lambda x: x[0],
        )[1]

    with col2:
        thalach = st.number_input("Max heart rate achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox(
            "Exercise induced angina?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0],
        )[1]
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of peak exercise ST segment",
            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0],
        )[1]
        ca = st.selectbox("Number of major vessels colored (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            options=[("Normal", 0), ("Fixed defect", 1), ("Reversible defect", 2)],
            format_func=lambda x: x[0],
        )[1]

    submitted = st.form_submit_button("Predict")

if submitted:
    patient = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    patient_df = pd.DataFrame([patient])
    proba_disease = pipeline.predict_proba(patient_df)[0, 1]
    label = int(proba_disease >= 0.5)
    verdict = "⚠️ Disease Likely" if label == 1 else "✅ No Disease Likely"

    st.subheader(verdict)
    st.metric("Estimated probability of heart disease", f"{proba_disease * 100:.1f}%")
    st.progress(min(max(proba_disease, 0.0), 1.0))
    st.caption("This is a machine learning estimate, not a medical diagnosis. Consult a doctor for real advice.")
