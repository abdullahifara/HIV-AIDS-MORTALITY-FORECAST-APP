
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HIV/AIDS Forecast", layout="wide")
st.title("HIV/AIDS Mortality Forecasting in Africa")
st.markdown("This app forecasts future HIV/AIDS deaths using ARIMA models.")

data_source = st.file_uploader("Upload your CSV file", type="csv")
if data_source is not None:
    data = pd.read_csv(data_source)
else:
    data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")

data['Year'] = pd.to_datetime(data['Year'])
data.set_index('Year', inplace=True)

train = data.loc[:'2013']
test = data.loc['2014':'2023']

model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(2, 2, 3))
model_fit = model.fit()

# Sidebar forecast year selection
max_forecast_year = 2030
forecast_end_year = st.sidebar.selectbox("Select forecast end year", list(range(2024, max_forecast_year + 1)))
forecast_steps = forecast_end_year - 2023
future_years = pd.date_range(start='2024', periods=forecast_steps, freq='Y')

forecast = model_fit.forecast(steps=forecast_steps)
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': forecast}, index=future_years)

# Forecast Tab with bar plot
st.markdown("### ARIMA Forecast ({}–{}) [Model trained up to 2013]".format(2024, forecast_end_year))
st.markdown("#### Forecast from 2024 to {} using ARIMA(2,2,3)".format(forecast_end_year))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train.index, train['Total_HIV_AIDS_Deaths'], label="Train", color="blue")
ax.plot(test.index, test['Total_HIV_AIDS_Deaths'], label="Test", color="orange")
sns.barplot(x=future_years.year, y=forecast, ax=ax, color="green", label="Forecast")
ax.legend()
st.pyplot(fig)

# Show forecasted values
if st.checkbox("📉 Forecasted Values:"):
    st.dataframe(forecast_df)

# Download CSV
csv = forecast_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False).encode()
st.download_button("Download Forecast CSV", data=csv, file_name="hiv_aids_forecast.csv", mime="text/csv")
