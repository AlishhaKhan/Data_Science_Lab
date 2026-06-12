import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Stroke Prediction by Alisha")
st.markdown("Provide the following details.")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_heart_rate = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'fastingBS': fasting_bs,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + rest_ecg: 1,
        'ST_Slope_' + st_slope: 1,
        'ExerciseAngina_' + str(exercise_angina): 1,
    } 
    
    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    selected_input = scaler.transform(input_df)
    prediction = model.predict(selected_input)
    
    if prediction ==1:
        st.error("High risk of heart stroke. Please consult a doctor immediately.")
    else:
        st.success("Low risk of heart stroke. Maintain a healthy lifestyle and regular check-ups.")        