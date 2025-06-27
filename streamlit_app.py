import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessing pipeline
model = joblib.load("model.pkl")              # Your trained classifier
preprocessor = joblib.load("preprocessor.pkl")  # Your preprocessing pipeline

st.title("💓 Heart Disease Risk Predictor")

st.write("Enter the patient details below to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == "Male" else 0],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal],
})

# Predict
if st.button("Predict"):
    try:
        X_processed = preprocessor.transform(input_data)
        prediction = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The patient is at risk of heart disease. (Risk Score: {proba:.2f})")
        else:
            st.success(f"✅ The patient is not at risk of heart disease. (Risk Score: {proba:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
