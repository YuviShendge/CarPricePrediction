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

# consider adding ,'condition' even if low on heat map
# used age bc it is the difference between the sell date and year and would be a better measurement
x = df[['mmr','odometer','year','make','model']] # Feature
y = df[['sellingprice']] # Target

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['make','model']) 
    ],
    remainder='passthrough'  # other columns unchanged
)

# This pipeline is used to preprocess data and train model together
model = Pipeline(steps=[
    ('preprocessor',preprocessor), ('regressor', LinearRegression())
])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# fake ramdom number allowing reproduction 

model.fit(x_train, y_train)
joblib.dump(model,'complexModel.pkl')
print("Model saved to complexModel.pkl")