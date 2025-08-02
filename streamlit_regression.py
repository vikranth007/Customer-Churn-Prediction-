import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# --- Load the trained model and preprocessing tools ---
model = tf.keras.models.load_model('regression_model.h5')

with open('regression_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('regression_ohe.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('regression_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 

# --- Streamlit UI ---
st.title("💼 Customer Churn Salary Prediction App")
st.subheader("Enter Customer Details:")

geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# --- Predict Button ---
st.button("Predict Salary")
    # --- Prepare the input data ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine encoded and numerical data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# --- Make prediction ---
prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

# --- Show result ---
st.success(f"💰 Predicted Estimated Salary: **${prediction_salary:,.2f}**")