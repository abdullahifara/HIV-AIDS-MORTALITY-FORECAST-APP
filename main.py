
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(page_title="HIV/AIDS Forecast", layout="wide")
st.title("HIV/AIDS Mortality Forecasting in Africa")
st.markdown("This app forecasts future HIV/AIDS deaths using ARIMA models.")

# Upload or use default CSV
data_source = st.file_uploader("Upload your CSV file", type="csv")
if data_source is not None:
    data = pd.read_csv(data_source)
else:
    data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")

# Preprocessing
data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-12-31')
data.set_index('Year', inplace=True)

# Forecast range
start_year = 2024
selected_end_year = st.sidebar.selectbox("Select forecast end year", list(range(2024, 2031)))
forecast_steps = selected_end_year - start_year + 1
future_years = pd.date_range(start=str(start_year), periods=forecast_steps, freq="Y")

# Split data
train = data.loc[:'2013']
test = data.loc['2014':'2023']

# Try auto_arima and fallback to ARIMA
try:
    from pmdarima import auto_arima
    model = auto_arima(train['Total_HIV_AIDS_Deaths'], seasonal=False, stepwise=True, suppress_warnings=True)
    model_fit = model
    forecast = model.predict(n_periods=forecast_steps)
    st.success("✅ Using auto_arima for forecasting.")
except Exception as e:
    st.warning("⚠️ auto_arima failed, using fallback ARIMA(3,2,2).")
    model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(3, 2, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)

# Build forecast DataFrame
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': forecast}, index=future_years)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EDA", "Forecast", "Residuals", "Download"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.tail())

with tab2:
    st.subheader("Exploratory Data Analysis")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data['Total_HIV_AIDS_Deaths'], marker='o')
    ax1.set_title("HIV/AIDS Deaths Over Time")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Deaths")
    st.pyplot(fig1)

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    sns.barplot(x=data.index.year, y=data['Total_HIV_AIDS_Deaths'], ax=ax_bar, palette="coolwarm")
    ax_bar.set_title("Bar Plot of HIV/AIDS Deaths by Year")
    ax_bar.set_xlabel("Year")
    ax_bar.set_ylabel("Total Deaths")
    plt.xticks(rotation=45)
    st.pyplot(fig_bar)

with tab3:
    st.subheader("ARIMA Forecast")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(train.index, train['Total_HIV_AIDS_Deaths'], label="Train", color="blue")
    ax2.plot(test.index, test['Total_HIV_AIDS_Deaths'], label="Test", color="orange")
    ax2.plot(forecast_df.index, forecast_df['Predicted_HIV_Deaths'], label="Forecast", linestyle="--", color="green", marker="o")
    ax2.set_xlim([train.index[0], forecast_df.index[-1]])
    ax2.set_title("Forecast ({}–{})".format(start_year, selected_end_year))
    ax2.legend()
    st.pyplot(fig2)
    st.write("📈 Forecasted Values:")
    st.dataframe(forecast_df)

with tab4:
    st.subheader("Model Residuals")
    if hasattr(model_fit, 'arima_res_'):
        residuals = train['Total_HIV_AIDS_Deaths'] - model_fit.arima_res_.fittedvalues
    else:
        residuals = train['Total_HIV_AIDS_Deaths'] - model_fit.fittedvalues
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(residuals)
    ax3.axhline(0, linestyle='--', color='red')
    ax3.set_title("Residuals")
    st.pyplot(fig3)

with tab5:
    st.subheader("Download Forecast as CSV")
    forecast_df.index.name = "Year"
    csv = forecast_df.reset_index().to_csv(index=False).encode()
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name='hiv_aids_forecast.csv',
        mime='text/csv'
    )
