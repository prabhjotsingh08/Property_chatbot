import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model(path="House_Prediction_Model.pkl"):
    return joblib.load(path)

model = load_model()

# Prediction function
def predict_price(location, sqft, bath, bhk, model):
    input_data = pd.DataFrame([[location, sqft, bath, bhk]],
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    return model.predict(input_data)[0]

# Streamlit UI
st.title("🏠 House Price Prediction App")

# Inputs
location = st.text_input("Enter Location", "Indira Nagar")
sqft = st.number_input("Enter Total Square Feet", min_value=100.0, step=50.0, value=1200.0)
bath = st.number_input("Enter Number of Bathrooms", min_value=1, step=1, value=2)
bhk = st.number_input("Enter Number of BHK", min_value=1, step=1, value=3)

if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk, model)
    st.success(f"🏡 Predicted Price: ₹ {price:,.2f}")
