import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "Cleaned_CarPrices.csv"
try:
    df = pd.read_csv(file_path)
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error(f"File not found at {file_path}. Please check the path and try again.")
    st.stop()

# Load models
try:
    complex_model = joblib.load("complexModel.pkl")
    basic_model = joblib.load("model.pkl")
    st.success("Models loaded successfully!")
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Dashboard Title
st.title("Car Price Prediction Dashboard")
st.write("This dashboard predicts car selling prices using basic and complex models.")

# Display Dataset
st.write("### Cleaned Car Prices Data")
st.dataframe(df.head())

# Regression Plot
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

# User Inputs for Prediction
st.subheader("Predict Selling Price")
mmr = st.number_input("Enter MMR: ", min_value=0.0, value=10000.0, key="mmr_input")
make = st.selectbox("Select Make", df['make'].unique(), key="make_input")
model_input = st.selectbox("Select Model", df[df['make'] == make]['model'].unique(), key="model_input")
odometer = st.number_input("Enter Odometer (mileage): ", min_value=0.0, value=50000.0, key="odometer_input")
year = st.number_input("Enter Year: ", min_value=1920, value=2025, key="year_input")
seller = st.selectbox("Select Seller", df['seller'].unique(), key="seller_input")

# Create Input Data
input_data = pd.DataFrame({
    'mmr': [mmr],
    'odometer': [odometer],
    'make': [make],
    'model': [model_input],
    'year': [year],
    'seller': [seller]
})

# Model Comparison
st.subheader("Model Comparison")
if st.button("Predict"):
    try:
        basic_prediction = basic_model.predict(input_data[['mmr', 'seller']])
        complex_prediction = complex_model.predict(input_data)

        st.write("### Prediction Results")
        st.write(f"**Basic Model Prediction:** ${float(basic_prediction[0]):,.2f}")
        st.write(f"**Complex Model Prediction:** ${float(complex_prediction[0]):,.2f}")

        # Visual Comparison
        comparison = pd.DataFrame({
            "Model": ["Basic Model", "Complex Model"],
            "Predicted Price": [float(basic_prediction[0]), float(complex_prediction[0])]
        })
        st.bar_chart(comparison.set_index("Model"))

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Display Metrics for Models
st.subheader("Model Performance Metrics")
basic_r2 = r2_score(y_regression, basic_model.predict(df[['mmr', 'seller']]))
complex_r2 = r2_score(y_regression, complex_model.predict(df[['mmr', 'odometer', 'make', 'model', 'year', 'seller']]))

st.write(f"**Basic Model R² Score:** {basic_r2:.2f}")
st.write(f"**Complex Model R² Score:** {complex_r2:.2f}")
