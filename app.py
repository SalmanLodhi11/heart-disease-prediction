
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Page title
st.title("Heart Disease Prediction System")
st.write("Enter the patient's information below.")

# Get input values from the user
age = st.number_input("Age", min_value=1, max_value=120, value=45)
trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=120
)
chol = st.number_input(
    "Serum Cholesterol (mg/dl)",
    min_value=50,
    max_value=700,
    value=200
)
thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

# Make a prediction when the button is clicked
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, trestbps, chol, thalach]],
        columns=['age', 'trestbps', 'chol', 'thalach']
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Heart disease detected by the model.")
    else:
        st.success("No heart disease detected by the model.")

st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)
