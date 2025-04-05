
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HIV/AIDS Forecast", layout="wide")

st.title("HIV/AIDS Mortality Forecasting in Africa")
st.markdown("This app forecasts future HIV/AIDS deaths using ARIMA(2,2,3) model.")

# Upload or use default CSV
data_file = st.file_uploader("Upload your CSV file", type="csv")
if data_file is not None:
    data = pd.read_csv(data_file)
else:
    data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")

# Convert 'Year' to datetime
data['Year'] = pd.to_datetime(data['Year'])
data.set_index('Year', inplace=True)

# Split data
train = data.loc[:'2013']
test = data.loc['2014':'2023']

# Forecasting range
forecast_end_year = st.sidebar.selectbox("Select forecast end year", list(range(2024, 2031)))
forecast_steps = forecast_end_year - 2023
start_year = 2024
future_years = pd.date_range(start=str(start_year), periods=forecast_steps, freq='Y')

# Fit model with ARIMA(2,2,3)
model = ARIMA(train["Total_HIV_AIDS_Deaths"], order=(2,2,3))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_steps)
forecast_df = pd.DataFrame({"Predicted_HIV_Deaths": forecast}, index=future_years)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EDA", "Forecast", "Residuals", "Download"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.tail())

with tab2:
    st.subheader("Exploratory Data Analysis")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data["Total_HIV_AIDS_Deaths"], marker='o')
    ax1.set_title("HIV/AIDS Deaths Over Time")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=data.index.year, y=data["Total_HIV_AIDS_Deaths"], ax=ax2, palette="coolwarm")
    ax2.set_title("Deaths by Year")
    st.pyplot(fig2)

with tab3:
    st.subheader(f"ARIMA Forecast ({start_year}–{forecast_end_year}) [Model trained up to 2013]")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(train.index, train["Total_HIV_AIDS_Deaths"], label="Train", color='blue')
    ax3.plot(test.index, test["Total_HIV_AIDS_Deaths"], label="Test", color='orange')
    sns.barplot(x=[str(y.year) for y in forecast_df.index], y=forecast_df["Predicted_HIV_Deaths"].values,
                ax=ax3, color="green", label="Forecast")
    ax3.set_title(f"Forecast from {start_year} to {forecast_end_year} using ARIMA(2,2,3)")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("HIV/AIDS Deaths")
    ax3.legend()
    st.pyplot(fig3)

    if st.checkbox("✅ Forecasted Values:"):
        st.dataframe(forecast_df)

with tab4:
    st.subheader("Model Residuals")
    residuals = train["Total_HIV_AIDS_Deaths"] - model_fit.fittedvalues
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(residuals)
    ax4.axhline(0, linestyle='--', color='red')
    ax4.set_title("Residuals")
    st.pyplot(fig4)

with tab5:
    st.subheader("Download Forecast")
    csv = forecast_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "hiv_forecast.csv", "text/csv")
