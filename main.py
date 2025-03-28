# %%
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
np.random.seed(42)
import pickle  # Import pickle to save the model
import random
np.random.seed(42)
random.seed(42)



# %%
data=pd.read_csv("/Users/abdullahifarahabdi/Desktop/DATA 2/cleaned_total_hiv_aids_deaths_africa.csv")
data.head()

# %%
data.tail()

# %%
data.info()

# %%
data.shape

# %% [markdown]
# # : Identify the year with the highest deaths

# %%

max_death_year = data.loc[data['Total_HIV_AIDS_Deaths'].idxmax()]
print(f"Year with highest deaths: {max_death_year['Year']}, Deaths: {max_death_year['Total_HIV_AIDS_Deaths']}")

# %%
plt.figure(figsize=(12, 6))
sns.barplot(x=data['Year'], y=data['Total_HIV_AIDS_Deaths'], palette="coolwarm")
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Total HIV/AIDS Deaths')
plt.title('Total HIV/AIDS Deaths per Year')
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Total_HIV_AIDS_Deaths'], marker='o', linestyle='-', color='b')
plt.xlabel('Year')
plt.ylabel('Total HIV/AIDS Deaths')
plt.title('HIV/AIDS Deaths Over the Years')
plt.grid(True)
plt.show()

# %%
# Sort data in descending order for better visualization
data_sorted = data.sort_values(by='Total_HIV_AIDS_Deaths', ascending=False)

# 📊 **Pie Chart for All Years (1990-2023)**
plt.figure(figsize=(10, 10))
plt.pie(
    data_sorted['Total_HIV_AIDS_Deaths'], 
    labels=data_sorted['Year'], 
    autopct='%1.1f%%',  # Display percentage values
    colors=sns.color_palette("coolwarm", len(data_sorted))
)
plt.title('Proportion of HIV/AIDS Deaths by Year (1990-2023)')
plt.show()

# %%
# : Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
plt.plot(data, marker='o', linestyle='-', color='b', label='Total HIV/AIDS Deaths')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('HIV/AIDS Deaths in Africa Over Time, Exploratory Data Analysis (EDA)')
plt.legend()
plt.show()

# %% [markdown]
# # Convert 'Year' column to datetime format with 31st December of each year

# %%
data['Year'] = pd.to_datetime(data['Year'].astype(str) + '-12-31', format='%Y-%m-%d')

# Set 'Year' as the index
data.set_index('Year', inplace=True)

# Verify the change
print(data.head())


# %%
data.head()


# %%
plt.figure(figsize=(12, 6))
plt.plot(data, marker='o', linestyle='-', color='b', label='Total HIV/AIDS Deaths')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('HIV/AIDS Deaths in Africa (1990-2023)')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Step 3: Check for Stationarity
def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is non-stationary. Differencing needed.")

adf_test(data['Total_HIV_AIDS_Deaths'])

# %%
# Step 6: Split Data into Training and Testing Sets
train = data.loc[:'2013']  # Training data from 1990 to 2015
test = data.loc['2014':'2023']  # Testing data from 2016 to 2023

# %%
# Import required libraries
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create a figure with two subplots (ACF on the left, PACF on the right)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the ACF (Autocorrelation Function)
plot_acf(train['Total_HIV_AIDS_Deaths'], ax=axes[0])  
axes[0].set_title("ACF Plot - Checking for Moving Average (q)")

# Plot the PACF (Partial Autocorrelation Function)
plot_pacf(train['Total_HIV_AIDS_Deaths'], ax=axes[1])  
axes[1].set_title("PACF Plot - Checking for Autoregression (p)")

# Show the plots
plt.show()


# %%
# Plot the ACF and PACF for the entire time series (not just train)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(data['Total_HIV_AIDS_Deaths'], ax=axes[0])  
axes[0].set_title("ACF Plot - Full Time Series")

plot_pacf(data['Total_HIV_AIDS_Deaths'], ax=axes[1])  
axes[1].set_title("PACF Plot - Full Time Series")

plt.show()


# %%
# Step 7: Use Auto ARIMA to Find Best Parameters

best_arima = auto_arima(
    train['Total_HIV_AIDS_Deaths'], 
    seasonal=False,  # No seasonality
    stepwise=True,   # Faster model selection
    trace=True       # Print progress of model selection
)

# Print the best model's summary
print(best_arima.summary())


# %%
plt.figure(figsize=(10, 5))
plt.plot(train, label='Training Data', marker='o', linestyle='-')
plt.plot(test, label='Testing Data', marker='o', linestyle='dashed')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Training vs Testing Data')
plt.legend()
plt.show()

# %%
# Step 8: Fit the Best ARIMA Model

from statsmodels.tsa.arima.model import ARIMA
import pickle  # Import pickle to save the model

# Train ARIMA model with the best parameters
model = ARIMA(train['Total_HIV_AIDS_Deaths'], order=(3, 2,2 ))
model_fit = model.fit()

# Save the trained model
with open("arima_model.pkl", "wb") as f:
    pickle.dump(model_fit, f)

print("✅ Model trained and saved successfully.")



# %%
forecast_steps = 7  # Forecast for 7 years from 2024 to 2030
start_year = 2024  # Start forecasting from 2024

# Extend data index to include missing years (2014-2023)
last_year = train.index[-1].year  # Get last year in training data (should be 2013)
years_to_extend = list(range(last_year + 1, start_year))  # Missing years (2014-2023)

# Generate future year index from 2024 to 2030
future_years = pd.date_range(start=str(start_year), periods=forecast_steps, freq='Y')

# ✅ Predict for 2014-2023 first, then 2024-2030
model_extended = model_fit.predict(start=len(train), end=len(train) + len(years_to_extend) + forecast_steps - 1)

# ✅ Remove 2014-2023, keeping only 2024-2030
future_forecast = model_extended[len(years_to_extend):]

# Convert forecast to DataFrame and assign correct index
forecast_df = pd.DataFrame({'Predicted_HIV_Deaths': future_forecast})
forecast_df.index = future_years  # Set index to future years

# Print the forecasted values
print(forecast_df)



# %%
# ✅ Visualizing Training, Testing, and Corrected Forecast
plt.figure(figsize=(10, 5))
plt.plot(train, label='Training Data', marker='o', linestyle='-')
plt.plot(test, label='Testing Data', marker='o', linestyle='dashed')
plt.plot(forecast_df, label='Forecasted Data (2024-2030)', marker='o', linestyle='dashed', color='r')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Training, Testing, and Forecasted Data (Corrected Forecast from 2024)')
plt.legend()
plt.show()

# %%
# Step 8: Model Diagnostics
model_fit.plot_diagnostics(figsize=(12, 8))
plt.show()

# %% [markdown]
# #performing a Ljung-Box test on residuals

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox


ljung_box_results = acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)

print(ljung_box_results)


# %% [markdown]
# #The ARIMA model appears to have captured all the meaningful patterns in the data, and there is no strong evidence of model misspecification.
#  #The p-value (0.842 > 0.05) indicates that the residuals are not significantly autocorrelated, confirming that the ARIMA model effectively captured the time-dependent structure of the data.
#  #The model residuals exhibit white noise behaviour, meaning no significant patterns remain, validating its ability to provide unbiased forecasts.
# 

# %%
from statsmodels.stats.diagnostic import het_arch

# Perform ARCH test for heteroscedasticity
arch_stat, arch_pvalue, _, _ = het_arch(model_fit.resid)

print(f"ARCH Test Statistic: {arch_stat:.4f}")
print(f"p-value: {arch_pvalue:.4f}")


# %% [markdown]
# #check residual

# %%
# Compute residuals for training data
train_residuals = train['Total_HIV_AIDS_Deaths'] - model_fit.fittedvalues

# Compute residuals for test data
test_forecast = model_fit.forecast(steps=len(test))
test_residuals = test['Total_HIV_AIDS_Deaths'] - test_forecast

# Combine residuals for full period (training + testing)
full_residuals = pd.concat([train_residuals, test_residuals])

# Plot residuals over time
plt.figure(figsize=(10, 5))
plt.plot(full_residuals, marker='o', linestyle='-', color='b', label='Residuals')
plt.axhline(y=0, color='red', linestyle='dashed')  # Reference line at zero
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residuals Over Time (Training & Testing Data)')
plt.legend()
plt.show()


# %%
# ✅ Ensure forecast_steps matches the number of predicted values
forecast_steps = len(future_forecast)

# ✅ Define the correct future years, ensuring the length matches future_forecast
future_years = list(range(2023, 2023 + forecast_steps))

# ✅ Convert forecasted values into a DataFrame
forecast_df = pd.DataFrame({'Year': future_years, 'Predicted_HIV_Deaths': future_forecast})

# ✅ Convert 'Year' to a string for better plotting
forecast_df['Year'] = forecast_df['Year'].astype(str)

# %%
# 📊 Bar Graph - Forecasted Data (2024-2030)
plt.figure(figsize=(12, 6))

# Filter forecast data to include only years from 2024 onwards
forecast_filtered = forecast_df[forecast_df.index.year >= 2024]

# Create bar plot
sns.barplot(x=forecast_filtered.index.year, y=forecast_filtered['Predicted_HIV_Deaths'], palette="coolwarm")

# Formatting
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Predicted HIV/AIDS Deaths')
plt.title('Forecasted HIV/AIDS Deaths (2024-2030)')
plt.show()


# %%
# 📈 Line Graph - Forecasted Data (2024-2030)
plt.figure(figsize=(12, 6))

# Filter forecast data to include only years from 2024 onwards
forecast_filtered = forecast_df[forecast_df.index.year >= 2024]

# Create line plot
plt.plot(forecast_filtered.index.year, forecast_filtered['Predicted_HIV_Deaths'], marker='o', linestyle='-', color='r')

# Formatting
plt.xlabel('Year')
plt.ylabel('Predicted HIV/AIDS Deaths')
plt.title('Trend of Forecasted HIV/AIDS Deaths (2024-2030)')
plt.grid(True)
plt.show()


# %%
# 🥧 Pie Chart - Forecasted Data Distribution (2024-2030)
plt.figure(figsize=(8, 8))

# Filter forecast data to include only years from 2024 onwards
forecast_filtered = forecast_df[forecast_df.index.year >= 2024]

# Create pie chart
plt.pie(forecast_filtered['Predicted_HIV_Deaths'], labels=forecast_filtered.index.year, autopct='%1.1f%%', 
        colors=sns.color_palette("coolwarm", len(forecast_filtered)))

# Formatting
plt.title('Proportion of Forecasted HIV/AIDS Deaths (2024-2030)')
plt.show()



# %%
# Generate predictions for the test period
test_forecast = model_fit.forecast(steps=len(test))

# Compute Error Metrics
mae = mean_absolute_error(test['Total_HIV_AIDS_Deaths'], test_forecast)
mse = mean_squared_error(test['Total_HIV_AIDS_Deaths'], test_forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test['Total_HIV_AIDS_Deaths'] - test_forecast) / test['Total_HIV_AIDS_Deaths'])) * 100
accuracy = 100 - mape

# Print Error Metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Model Accuracy: {accuracy:.2f}%")


