
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HIV/AIDS Forecast", layout="wide")
st.title("HIV/AIDS Mortality Forecasting in Africa")

# Upload or use default CSV
data_source = st.file_uploader("Upload your CSV file", type="csv")
if data_source is not None:
    data = pd.read_csv(data_source)
else:
    data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")

data['Year'] = pd.to_datetime(data['Year'])
data.set_index('Year', inplace=True)

# Sidebar: Forecast end year selection
min_forecast_year = 2024
max_forecast_year = 2030
forecast_end_year = st.sidebar.selectbox("Select forecast end year", list(range(min_forecast_year, max_forecast_year + 1)))
forecast_steps = forecast_end_year - 2023

# Train/Test Split
train = data.loc[:'2013']
test = data.loc['2014':'2023']

# Fit ARIMA(2,2,3)
model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(2, 2, 3))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start="2024-12-31", periods=forecast_steps, freq='Y')
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': forecast.values}, index=forecast_index)

# Display plot
st.subheader("ARIMA Forecast ({}–{}) [Model trained up to 2013]".format(min_forecast_year, forecast_end_year))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train.index, train['Total_HIV_AIDS_Deaths'], label="Train", color="blue")
ax.plot(test.index, test['Total_HIV_AIDS_Deaths'], label="Test", color="orange")
ax.plot(forecast_df.index, forecast_df["Predicted_HIV_Deaths"], 'go--', label="Forecast")
ax.set_title(f"Forecast from 2024 to {forecast_end_year} using ARIMA(2,2,3)")
ax.legend()
st.pyplot(fig)

# ✅ Bar Plot of Forecasted Values Only
st.subheader("Forecasted Values:")
fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
bar_x = [str(d.year) for d in forecast_df.index]
bar_y = forecast_df["Predicted_HIV_Deaths"].values
sns.barplot(x=bar_x, y=bar_y, ax=ax_bar, color="skyblue")
ax_bar.set_title("Forecasted HIV/AIDS Deaths by Year")
ax_bar.set_xlabel("Year")
ax_bar.set_ylabel("Predicted Deaths")
plt.xticks(rotation=45)
st.pyplot(fig_bar)

# Optional table display
if st.checkbox("Show forecasted data"):
    st.dataframe(forecast_df)
