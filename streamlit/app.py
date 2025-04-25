import streamlit as st
import requests
import json

# Flask API URL (replace with your deployed Flask API URL after deployment)
API_URL = "http://localhost:5000/predict" # Update to https://your-app.onrender.com/predict

st.title("Mental Health Depression Level Prediction")

# Input form
st.header("Enter User Data")
with st.form("prediction_form"):
    # Clinical Data
    st.subheader("Clinical Data")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    family_history = st.selectbox("Family History of Depression", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    no_employees = st.selectbox("Number of Employees", options=[0, 1, 2], format_func=lambda x: ["1-10", "11-50", "50+"][x])

    # Physiological Data
    st.subheader("Physiological Data")
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.5, step=0.1)
    quality_of_sleep = st.number_input("Quality of Sleep (1-10)", min_value=1, max_value=10, value=8)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=8000)
    sleep_quality_index = sleep_duration * quality_of_sleep

    # Text Data
    st.subheader("Text Data")
    social_media = st.text_area("Social Media Post", value="Feeling okay today, just chilling")
    chatbot = st.text_area("Chatbot Interaction", value="I’m doing alright, how about you?")

    # Optional Ground Truth
    depression_level = st.selectbox("Actual Depression Level (Optional)", options=[None, 0, 1, 2, 3], format_func=lambda x: "N/A" if x is None else ["None", "Mild", "Moderate", "Severe"][x])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Handle prediction
if submitted:
    payload = {
        "clinical": [age, gender, family_history, no_employees],
        "physiological": [sleep_duration, quality_of_sleep, heart_rate, daily_steps, sleep_quality_index],
        "social_media": social_media,
        "chatbot": chatbot,
        "depression_level": depression_level
    }
    try:
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successful!")
            st.write(f"**Full Fusion Prediction**: {result['depression_levels'][result['full_fusion_prediction']]}")
            st.write(f"**BERT-Only Prediction**: {result['depression_levels'][result['bert_only_prediction']]}")
            st.write(f"**Is Anomaly?**: {'Yes' if result['is_anomaly'] else 'No'}")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")

# Depression level legend
st.markdown("### Depression Levels")
st.write("0: None, 1: Mild, 2: Moderate, 3: Severe")