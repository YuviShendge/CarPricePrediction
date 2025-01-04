import pandas as pd
df = pd.read_csv('car_prices.csv')
# sets df = the data

# replaces blank fields with NA then drops any line with NA
df_cleaned = df.replace('',pd.NA,inplace=True)
df_cleaned = df.dropna()
df_cleaned.to_csv("Cleaned_CarPrices.csv", index=False)
# data das been cleaned and put in cleaned_data.csv
print("Cleaned data has been saved to Cleaned_CarPrices.csv")


