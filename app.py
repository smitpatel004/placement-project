import streamlit as st
import pickle
import numpy as np

with open('modelw.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("Student Placement Prediction App")

cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
iq = st.number_input("Enter your IQ", min_value=50, max_value=200, value=100, step=1)

if st.button("Predict"):
    input_features = np.array([[cgpa, iq]])
    input_features_scaled = scaler.transform(input_features) 
    prediction = model.predict(input_features_scaled)

    if prediction[0] == 1:
        st.success("The student is predicted to get placed!")
    else:
        st.error("The student is predicted NOT to get placed.")
