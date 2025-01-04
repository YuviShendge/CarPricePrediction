import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = "Cleaned_CarPrices.csv"
try:
    df = pd.read_csv(file_path)
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error(f"File not found at {file_path}. Please check the path and try again.")

# Load models
try:
    complex_model = joblib.load("complexModel.pkl")  
    st.success("Models loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")

# Dashboard Title
st.title("Car Price Prediction ML Dashboard")
st.write("This dashboard predicts car selling prices using machine learning models.")

# Display dataset
st.write("### Cleaned Car Prices Data")
st.dataframe(df.head())

#  Regression Plot
st.subheader("MMR vs Selling Price with Linear Regression")
x_regression = df['mmr'].values.reshape(-1, 1)
y_regression = df['sellingprice'].values
reg_model = LinearRegression()
reg_model.fit(x_regression, y_regression)
y_pred = reg_model.predict(x_regression)

plt.figure(figsize=(10, 6))
plt.scatter(df['mmr'], df['sellingprice'], alpha=0.7, color='blue', label='Data')
plt.plot(df['mmr'], y_pred, color='red', linestyle='--', label='Regression Line')
plt.legend()
plt.title('MMR vs Selling Price')
plt.xlabel('MMR')
plt.ylabel('Selling Price')
st.pyplot(plt)

# Basic Model
st.subheader("Basic Prediction Model")
mmr_basic = st.number_input("Enter MMR: ", min_value=0.0, value=10000.0, key="basic_mmr")
seller = st.selectbox("Select Seller", df['seller'].unique(), key="basic_seller")

if st.button("Predict Basic Model", key="predict_basic"):
    try:
        input_data_basic = pd.DataFrame({'mmr': [mmr_basic], 'seller': [seller]})
        prediction_basic = basic_model.predict(input_data_basic)
        st.success(f"Predicted Selling Price: ${float(prediction_basic[0]):,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Improved Prediction
st.subheader("Complex Prediction Model")
mmr_improved = st.number_input("Enter MMR: ", min_value=0.0, value=10000.0, key="improved_mmr")
make = st.selectbox("Select Make", df['make'].unique(), key="improved_make")
model_input = st.selectbox("Select Model", df[df['make'] == make]['model'].unique(), key="improved_model")
odometer = st.number_input("Enter Odometer (mileage): ", min_value=0.0, value=50000.0, key="improved_odometer")

if st.button("Predict Improved Model", key="predict_improved"):
    try:
        input_data_improved = pd.DataFrame({
            'mmr': [mmr_improved],
            'odometer': [odometer],
            'make': [make],
            'model': [model_input]
        })
        prediction_improved = complex_model.predict(input_data_improved)
        st.success(f"Predicted Selling Price: ${float(prediction_improved[0]):,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
