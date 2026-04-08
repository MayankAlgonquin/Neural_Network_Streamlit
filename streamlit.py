import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

import logging
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Title
st.title("Loan Eligibility Predictor")
st.write("""
This app predicts whether an applicant will be approved loan based on economic and demographic factors.
""")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
# Load model
try:
    rf_pickle = open("models/lrmodel.pkl", "rb")
    rf_model = pickle.load(rf_pickle)
    rf_pickle.close()

    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Form
with st.form("user_inputs"):
    st.subheader("Loan Prediction Inputs")

    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    submitted = st.form_submit_button("Predict")

if submitted:


    input_dict = {
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,

        'Gender_Female': 1 if Gender == "Female" else 0,
        'Gender_Male': 1 if Gender == "Male" else 0,

        'Married_No': 1 if Married == "No" else 0,
        'Married_Yes': 1 if Married == "Yes" else 0,

        'Dependents_0': 1 if Dependents == "0" else 0,
        'Dependents_1': 1 if Dependents == "1" else 0,
        'Dependents_2': 1 if Dependents == "2" else 0,
        'Dependents_3+': 1 if Dependents == "3+" else 0,

        'Education_Graduate': 1 if Education == "Graduate" else 0,
        'Education_Not Graduate': 1 if Education == "Not Graduate" else 0,

        'Self_Employed_No': 1 if Self_Employed == "No" else 0,
        'Self_Employed_Yes': 1 if Self_Employed == "Yes" else 0,

        'Property_Area_Rural': 1 if Property_Area == "Rural" else 0,
        'Property_Area_Semiurban': 1 if Property_Area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if Property_Area == "Urban" else 0,
    }

    input_df = pd.DataFrame([input_dict])


    with open("models/columns.pkl", "rb") as f:
        cols = pickle.load(f)

    input_df = input_df.reindex(columns=cols, fill_value=0)

   
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    input_scaled = scaler.transform(input_df)


    prediction = rf_model.predict(input_scaled)
    proba = rf_model.predict_proba(input_scaled)

    result = "Approved" if prediction[0] == 1 else "Not Approved"
    st.write(f"Loan Status: {result}")
  

st.write("""
We used a Logistic Regression model to predict the loan elgibility.
""")


