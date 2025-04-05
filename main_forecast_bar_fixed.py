
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="HIV/AIDS Forecast", layout="wide")
st.title("HIV/AIDS Mortality Forecasting in Africa")

# Load data
data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")
data['Year'] = pd.to_datetime(data['Year'])
data.set_index('Year', inplace=True)

# Train-test split
train = data.loc[:'2013']
test = data.loc['2014':'2023']

# ARIMA model
model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(2, 2, 3))
model_fit = model.fit()

# Forecast
forecast_years = st.sidebar.selectbox("Select forecast end year", list(range(2024, 2031)), index=3)
forecast_steps = forecast_years - 2023
future_years = pd.date_range(start="2024", periods=forecast_steps, freq='Y')
forecast = model_fit.forecast(steps=forecast_steps)

# Forecast Plot as Bar Graph
st.subheader(f"ARIMA Forecast (2024–{forecast_years}) [Model trained up to 2013]")
fig, ax = plt.subplots()
ax.plot(train.index, train['Total_HIV_AIDS_Deaths'], label='Train', color='blue')
ax.plot(test.index, test['Total_HIV_AIDS_Deaths'], label='Test', color='orange')
sns.barplot(x=[str(y.year) for y in future_years], y=forecast, ax=ax, color="green", label="Forecast")
ax.set_title(f"Forecast from 2024 to {forecast_years} using ARIMA(2,2,3)")
ax.legend()
st.pyplot(fig)

# Forecast Table
forecast_df = pd.DataFrame({'Year': future_years, 'Predicted_HIV_Deaths': forecast.values})
forecast_df.set_index('Year', inplace=True)

if st.checkbox("📉 Forecasted Values:"):
    st.dataframe(forecast_df)
