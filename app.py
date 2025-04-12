import streamlit as st
from disease_predictor import predict_disease

st.title("Medical Diagnoser 🏥")
st.write("Enter your symptoms below, and the app will predict the most likely disease.")

# User Input
user_input = st.text_input("Describe your symptoms:")

if user_input:
    predicted_disease, extracted_symptoms = predict_disease(user_input)
    
    st.write(f"**Extracted Symptoms:** {', '.join(extracted_symptoms)}")
    st.write(f"**Predicted Disease:** {predicted_disease}")
