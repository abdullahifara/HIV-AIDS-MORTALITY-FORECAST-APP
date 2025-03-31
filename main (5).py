
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
st.markdown("This app forecasts future HIV/AIDS deaths using ARIMA(3,2,2) — notebook-aligned with year selection.")

# Upload or load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")

# Prepare data
data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-12-31')
data.set_index('Year', inplace=True)

# Sidebar forecast range selection
start_year = 2024
selected_end_year = st.sidebar.selectbox("Select forecast end year", list(range(2024, 2031)))
forecast_steps = selected_end_year - start_year + 1
forecast_index = pd.date_range(start=f"{start_year}-12-31", periods=forecast_steps, freq="Y")

# Fit ARIMA(3,2,2) on full data
model = ARIMA(data['Total_HIV_AIDS_Deaths'], order=(3, 2, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=forecast_steps)
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': forecast.values}, index=forecast_index)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EDA", "Forecast", "Residuals", "Download"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.tail())

with tab2:
    st.subheader("Exploratory Data Analysis")
    fig1, ax1 = plt.subplots()
    data['Total_HIV_AIDS_Deaths'].plot(ax=ax1)
    ax1.set_title("HIV/AIDS Deaths Over Time")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.barplot(x=data.index.year, y=data['Total_HIV_AIDS_Deaths'], ax=ax2)
    ax2.set_title("Bar Chart of Deaths by Year")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

with tab3:
    st.subheader(f"ARIMA Forecast ({start_year}–{selected_end_year})")
    fig3, ax3 = plt.subplots()
    data['Total_HIV_AIDS_Deaths'].plot(ax=ax3, label="Observed", color="blue")
    forecast_df['Predicted_HIV_Deaths'].plot(ax=ax3, label="Forecast", linestyle="--", marker='o', color="green")
    ax3.set_title(f"Forecast from {start_year} to {selected_end_year} using ARIMA(3,2,2)")
    ax3.legend()
    st.pyplot(fig3)
    st.dataframe(forecast_df)

with tab4:
    st.subheader("Model Residuals")
    residuals = model_fit.resid
    fig4, ax4 = plt.subplots()
    residuals.plot(ax=ax4)
    ax4.axhline(0, linestyle='--', color='red')
    ax4.set_title("Residuals")
    st.pyplot(fig4)

with tab5:
    st.subheader("Download Forecast as CSV")
    csv = forecast_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False).encode()
    st.download_button("Download Forecast CSV", data=csv, file_name='hiv_forecast.csv', mime='text/csv')
