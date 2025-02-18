import streamlit as st
import pickle
import numpy as np


model_path=r"C:\Users\Smile\Downloads\python\sristi\stress_model.pkl"
# Load Model
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)


st.title("Human Stress Detection System")
st.write("Enter physiological data to predict stress levels.")


snoring_range = st.number_input("Snoring Range", min_value=0, max_value=10)  
respiration_rate = st.number_input("Respiration Rate", min_value=5, max_value=40)  
temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0) 
limb_movement = st.number_input("Limb Movement Rate", min_value=0, max_value=10)  
blood_oxygen = st.number_input("Blood Oxygen Level (SpO2)", min_value=70, max_value=100) 
eye_movement = st.number_input("Eye Movement", min_value=0, max_value=10)  
sleep_hours = st.number_input("Number of Hours of Sleep", min_value=0, max_value=12)  
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200) 

# Predict Stress Level
if st.button("Predict Stress Level"):

    input_data = np.array([[snoring_range, respiration_rate, temperature, limb_movement,
                            blood_oxygen, eye_movement, sleep_hours, heart_rate]])
    
    
    if input_data.shape[1] == 8:
        prediction = model.predict(input_data)
        stress_level_map = {
            0: "Low Stress",
            1: "Moderate Stress",
            2: "High Stress",
            3: "Very High Stress",
            4: "Extreme Stress"
        }
        
        predicted_label = stress_level_map.get(prediction[0], "Unknown Stress Level")
        st.write(f"Predicted Stress Level: {predicted_label}")
    else:
        st.error("Error: Incorrect number of features provided!")