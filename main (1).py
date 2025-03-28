
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import pickle
import random

np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore")

# Load Data
data = pd.read_csv("cleaned_total_hiv_aids_deaths_africa.csv")
data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-12-31')
data.set_index('Year', inplace=True)

# ADF Test
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print("Stationary" if result[1] <= 0.05 else "Non-stationary")

adf_test(data['Total_HIV_AIDS_Deaths'])

# Train-test split
train = data.loc[:'2013']
test = data.loc['2014':'2023']

# ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(train['Total_HIV_AIDS_Deaths'], ax=axes[0])
plot_pacf(train['Total_HIV_AIDS_Deaths'], ax=axes[1])
plt.show()

# Train ARIMA manually based on ACF/PACF or try common configs
model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(3, 2, 2))
model_fit = model.fit()

with open("arima_model.pkl", "wb") as f:
    pickle.dump(model_fit, f)

print("\nModel trained and saved successfully.")

# Forecast future
forecast_steps = 7
start_year = 2024
future_years = pd.date_range(start=str(start_year), periods=forecast_steps, freq='Y')
model_extended = model_fit.predict(start=len(train), end=len(train) + forecast_steps - 1)
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': model_extended})
forecast_df.index = future_years

# Visualizations
plt.figure(figsize=(10, 5))
plt.plot(train, label='Training')
plt.plot(test, label='Testing')
plt.plot(forecast_df, label='Forecast 2024-2030', linestyle='dashed')
plt.legend()
plt.title('Training, Testing, and Forecasted HIV/AIDS Deaths')
plt.show()

# Model diagnostics
model_fit.plot_diagnostics(figsize=(12, 8))
plt.show()

# Ljung-Box Test
print("\nLjung-Box Test:\n", acorr_ljungbox(model_fit.resid, lags=[10], return_df=True))

# ARCH Test
arch_stat, arch_pvalue, _, _ = het_arch(model_fit.resid)
print(f"\nARCH Test Statistic: {arch_stat:.4f}, p-value: {arch_pvalue:.4f}")

# Residual Analysis
train_residuals = train['Total_HIV_AIDS_Deaths'] - model_fit.fittedvalues
test_forecast = model_fit.forecast(steps=len(test))
test_residuals = test['Total_HIV_AIDS_Deaths'] - test_forecast
full_residuals = pd.concat([train_residuals, test_residuals])

plt.figure(figsize=(10, 5))
plt.plot(full_residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Over Time')
plt.legend()
plt.show()

# Forecast evaluation
mae = mean_absolute_error(test['Total_HIV_AIDS_Deaths'], test_forecast)
mse = mean_squared_error(test['Total_HIV_AIDS_Deaths'], test_forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test['Total_HIV_AIDS_Deaths'] - test_forecast) / test['Total_HIV_AIDS_Deaths'])) * 100
accuracy = 100 - mape

print(f"\nMAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, Accuracy: {accuracy:.2f}%")
