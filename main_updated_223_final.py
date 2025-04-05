
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
st.markdown("This app forecasts future HIV/AIDS deaths using ARIMA(2,2,3) trained on data up to 2013 (to match notebook logic).")

# Upload or load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")

# Prepare data
data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-12-31')
data.set_index('Year', inplace=True)

# Split into train and test
train = data.loc[:'2013']
test = data.loc['2014':'2023']

# Forecast config
start_year = 2024
selected_end_year = st.sidebar.selectbox("Select forecast end year", list(range(2024, 2031)))
forecast_years = list(range(2024, selected_end_year + 1))
forecast_steps = len(test) + len(forecast_years)
future_index = pd.date_range(start="2014-12-31", periods=forecast_steps, freq="Y")

# Fit ARIMA(3,2,2) on training data only
model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(2, 2, 3))
model_fit = model.fit()

# Predict from 2014 to selected_end_year
predictions = model_fit.predict(start=len(train), end=len(train) + forecast_steps - 1)
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': predictions.values}, index=future_index)

# Extract only forecast years to display
forecast_display = forecast_df.loc[forecast_df.index.year.isin(forecast_years)]

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
    st.subheader(f"ARIMA Forecast ({start_year}–{selected_end_year}) [Model trained up to 2013]")
    fig3, ax3 = plt.subplots()
    train['Total_HIV_AIDS_Deaths'].plot(ax=ax3, label="Train", color="blue")
    test['Total_HIV_AIDS_Deaths'].plot(ax=ax3, label="Test", color="orange")
    forecast_display['Predicted_HIV_Deaths'].plot(ax=ax3, label="Forecast", linestyle="--", marker='o', color="green")
    ax3.set_title(f"Forecast from {start_year} to {selected_end_year} using ARIMA(2,2,3)")
    ax3.legend()
    st.pyplot(fig3)
    st.dataframe(forecast_display)

with tab4:
    st.subheader("Model Residuals (Train Only)")
    residuals = model_fit.resid
    fig4, ax4 = plt.subplots()
    residuals.plot(ax=ax4)
    ax4.axhline(0, linestyle='--', color='red')
    ax4.set_title("Residuals")
    st.pyplot(fig4)

with tab5:
    st.subheader("Download Forecast as CSV")
    csv = forecast_display.reset_index().rename(columns={"index": "Year"}).to_csv(index=False).encode()
    st.download_button("Download Forecast CSV", data=csv, file_name='hiv_forecast.csv', mime='text/csv')
