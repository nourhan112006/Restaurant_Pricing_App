import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("restaurant_pricing_model.pkl")
category_encoder = joblib.load("category_encoder.pkl")
subcategory_encoder = joblib.load("subcategory_encoder.pkl")

st.set_page_config(page_title="Restaurant Pricing Engine")

st.title("🍽️ Restaurant Pricing Engine")
st.write("Predict the recommended selling price for a menu item.")

# User Inputs
cost = st.number_input("Cost", min_value=0.0, value=50.0)

count = st.number_input(
    "Historical Sales Count",
    min_value=0,
    value=10
)

category = st.selectbox(
    "Category",
    category_encoder.classes_
)

subcategory = st.selectbox(
    "Sub Category",
    subcategory_encoder.classes_
)

if st.button("Predict Price"):

    category_encoded = category_encoder.transform([category])[0]
    subcategory_encoded = subcategory_encoder.transform([subcategory])[0]

    input_data = pd.DataFrame({
        "Cost": [cost],
        "Count": [count],
        "Category": [category_encoded],
        "Sub Category": [subcategory_encoded]
    })

    prediction = model.predict(input_data)

    st.success(f"Recommended Selling Price: {prediction[0]:.2f}")