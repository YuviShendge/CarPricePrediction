import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('Cleaned_CarPrices.csv')

x = df[['mmr','seller']] # Feature
y = df[['sellingprice']] # Target

# Using one-hot encoding from scikit to make  'seller' numerical
# Column Transformer allow for one-hot to work with numerical data

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['seller']) 
    ],
    remainder='passthrough'  # Leave other columns unchanged
)

# This pipeline is used to preprocess data and train model together
model = Pipeline(steps=[
    ('preprocessor',preprocessor), ('regressor', LinearRegression())
])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# random state is basically a fake random number allowing reproduction each time the code is run

model.fit(x_train, y_train)
joblib.dump(model,'model.pkl')
print("Model saved to model.pkl")
