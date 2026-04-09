import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

import logging
import os
 #Implementing logger to track streamlit functions
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Title
st.title("College Application Predictor")
st.write("""
This app predicts whether an applicant will be approved for an admission based on academic factors.
""")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
# Load model
try:
    rf_pickle = open("models/mlpmodel.pkl", "rb")
    rf_model = pickle.load(rf_pickle)
    rf_pickle.close()

    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Form
with st.form("user_inputs"):
    st.subheader("Admission Prediction Inputs")

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340, value=320)
    with col2:
        LOR = st.selectbox("Letter of Recommendation (LOR)", [1.0, 2.0, 3.0, 4.0, 5.0])

    # Row 2
    col1, col2 = st.columns(2)
    with col1:
        TOEFL_Score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
    with col2:
        CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)

    # Row 3
    col1, col2 = st.columns(2)
    with col1:
        University_Rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    with col2:
        Research = st.selectbox("Research Experience", ["Yes", "No"])

    # Row 4
    SOP = st.selectbox("Statement of Purpose (SOP)", [1.0, 2.0, 3.0, 4.0, 5.0])

    submitted = st.form_submit_button("Predict Admission Class")

if submitted:
    
    #Create an input dictionary

    input_dict = {
        "GRE_Score": GRE_Score,
        "TOEFL_Score": TOEFL_Score,
        "University_Rating": University_Rating,
        "SOP": SOP,
        "LOR": LOR,
        "CGPA": CGPA,
        "Research": 1 if Research == "Yes" else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Align with training columns 
    with open("models/columns.pkl", "rb") as f:
        cols = pickle.load(f)

    input_df = input_df.reindex(columns=cols, fill_value=0)

    # Scale to ensure it is consistent with the trained model
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    #transform the scaled model
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = rf_model.predict(input_scaled)
    proba = rf_model.predict_proba(input_scaled)

    #Prints the result.
    result = "Admit" if prediction[0] == 1 else "Reject"
    st.write(f"Prediction: {result}")
    st.write(f"Probability: {proba[0][1]:.2f}")

    st.write("""
    We used a neural network to predict if you will get the admission.
    """)


#Appends image in the webpage
    st.image("feature_importance.png")