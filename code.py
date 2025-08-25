# --- Imports ---
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
from datetime import datetime  # For date and time manipulation
import csv  # For handling CSV files
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import defaultdict

# Load and preprocess the data
def load_data(file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Select only the columns we need: "datetime" and "tempmax" (maximum temperature)
    data = data[['datetime', 'tempmax']]

    # Convert the 'datetime' column to pandas datetime format, and handle errors by converting invalid dates to NaT
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')

    # Sort the data by 'datetime' to ensure it's in chronological order
    data.sort_values(by='datetime', inplace=True)

    # Reset the DataFrame index after sorting to maintain a clean index
    data.reset_index(drop=True, inplace=True)

    # Print out the first few rows
    print("Data loaded successfully.")
    print(data.head())  # Show the first few rows
    print(data.info())  # Display data types and null values

    return data

def remove_outliers(data, column_name):



    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')

    # Calculate Q1, Q3, and IQR for the column
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    print(f"Number of outliers in '{column_name}': {len(outliers)}")

    # Remove outliers
    data_cleaned = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]


    return data_cleaned, len(outliers)

def separate_yearly_data(data):
    data_2021 = [row for row in data if '2021' in row['datetime']]
    data_2022 = [row for row in data if '2022' in row['datetime']]
    return data_2021, data_2022

# Calculate Mean Squared Error (MSE)
def mean_squared_error(actual, predicted):
    # Initialize the sum of squared errors
    squared_errors_sum = 0


    for a, p in zip(actual, predicted):

        squared_error = (a - p) ** 2

        squared_errors_sum += squared_error

    # Calculate the mean of the squared errors
    mse = squared_errors_sum / len(actual)
    return mse

def mean_absolute_deviation(actual, predicted):

    absolute_deviations_sum = 0

    for a, p in zip(actual, predicted):

        absolute_deviation = abs(a - p)

        absolute_deviations_sum += absolute_deviation

    mad = absolute_deviations_sum / len(actual)
    return mad

def extract_first_day_data_for_year(file_path, year=2021):
    try:
        df = pd.read_csv(file_path)


        if 'datetime' not in df.columns:
            raise ValueError(f"'datetime' column not found in {file_path}")


        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        df = df.dropna(subset=['datetime'])

        df.set_index('datetime', inplace=True)
        first_days = df.resample('MS').first()

        # Filter for the specified year
        return first_days[first_days.index.year == year]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Function to perform simple exponential smoothing
def exponential_smoothing_12(data, alpha):
    if not data:
        return []

    # Initialize the forecast list with the first value
    smoothed = [data[0]]


    for t in range(1, len(data)):
        forecast = alpha * data[t] + (1 - alpha) * smoothed[-1]
        smoothed.append(forecast)

    return smoothed


def calculate_5day_moving_average(data, year):

    forecasted_data = []
    for i in range(5, len(data)):
        tempmax_sum = 0
        if i < 10:
            for j in range(i - 4, i + 1):
                tempmax_sum += float(data[j]['tempmax'])
        else:
            for j in range(i - 5, i):
                if year == 2021 and data[j]['datetime'].startswith('2021-01'):
                    tempmax_sum += float(data[j]['tempmax'])
                elif year == 2022 and data[j]['datetime'].startswith('2022-01'):
                    tempmax_sum += float(data[j]['tempmax'])
                else:
                    tempmax_sum += forecasted_data[j - 5]['forecasted_tempmax']
        moving_average = tempmax_sum / 5
        forecasted_data.append({'datetime': data[i]['datetime'], 'forecasted_tempmax': moving_average})
    return forecasted_data

# --- Function to calculate 7-day moving average ---
def calculate_7day_moving_average(data, year):

    forecasted_data = []
    for i in range(7, len(data)):
        tempmax_sum = 0
        if i < 14:
            for j in range(i - 6, i + 1):
                tempmax_sum += float(data[j]['tempmax'])
        else:
            for j in range(i - 7, i):
                if year == 2021 and data[j]['datetime'].startswith('2021-01'):
                    tempmax_sum += float(data[j]['tempmax'])
                elif year == 2022 and data[j]['datetime'].startswith('2022-01'):
                    tempmax_sum += float(data[j]['tempmax'])
                else:
                    tempmax_sum += forecasted_data[j - 7]['forecasted_tempmax']
        moving_average = tempmax_sum / 7
        forecasted_data.append({'datetime': data[i]['datetime'], 'forecasted_tempmax': moving_average})
    return forecasted_data

def calculate_7day_weighted_moving_average(data, year):


    forecasted_data = []
    weights = [1/28, 2/28, 3/28, 4/28, 5/28, 6/28, 7/28]

    # Loop through the data, starting from the 8th day
    for i in range(7, len(data)):
        weighted_sum = 0

        # For the first 7 days of February, use only actual values
        if i < 14:  # February starts from index 7 (for simplicity)
            for j in range(i - 6, i + 1):
                weighted_sum += float(data[j]['tempmax']) * weights[j - (i - 6)]
        else:
            # For subsequent days, mix actual values and forecasted values
            for j in range(i - 7, i):
                weight = weights[j - (i - 7)]
                # Ensure that we are using actual data up until January
                if year == 2021 and data[j]['datetime'].startswith('2021-01'):
                    weighted_sum += float(data[j]['tempmax']) * weight
                elif year == 2022 and data[j]['datetime'].startswith('2022-01'):
                    weighted_sum += float(data[j]['tempmax']) * weight
                else:
                    # Use the forecasted value from the previous 7 days for the moving average
                    weighted_sum += forecasted_data[j - 7]['forecasted_tempmax'] * weight

        # Calculate the weighted moving average for the day
        moving_average = weighted_sum
        forecasted_data.append({'datetime': data[i]['datetime'], 'forecasted_tempmax': moving_average})

    return forecasted_data

def calculate_5day_weighted_moving_average(data, year):

    forecasted_data = []
    weights = [1/15, 2/15, 3/15, 4/15, 5/15]

    # Loop through the data, starting from the 6th day
    for i in range(5, len(data)):
        weighted_sum = 0

        # For the first 5 days of February
        if i < 10:
            for j in range(i - 4, i + 1):
                weighted_sum += float(data[j]['tempmax']) * weights[j - (i - 4)]
        else:

            for j in range(i - 5, i):
                weight = weights[j - (i - 5)]

                if year == 2021 and data[j]['datetime'].startswith('2021-01'):
                    weighted_sum += float(data[j]['tempmax']) * weight
                elif year == 2022 and data[j]['datetime'].startswith('2022-01'):
                    weighted_sum += float(data[j]['tempmax']) * weight
                else:
                    # Use the forecasted value from the previous 5 days for the moving average
                    weighted_sum += forecasted_data[j - 5]['forecasted_tempmax'] * weight


        moving_average = weighted_sum
        forecasted_data.append({'datetime': data[i]['datetime'], 'forecasted_tempmax': moving_average})

    return forecasted_data

# Exponential Smoothing calculation
def exponential_smoothing(actual, alpha):
    forecast = [actual[0]]
    for t in range(1, len(actual)):
        ft = alpha * actual[t - 1] + (1 - alpha) * forecast[-1]
        forecast.append(ft)
    return forecast

# Find the best smoothing parameter alpha based on January data
def find_best_alpha(jan_actual, rest_actual, alpha_values):
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]  # Define specific alpha values to test
    best_alpha_mse = None
    best_mse = float('inf')
    best_alpha_mad = None
    best_mad = float('inf')
    best_forecast_mse = []
    best_forecast_mad = []

    for alpha in alphas:

        forecast = exponential_smoothing(jan_actual, alpha)

        # Extend the forecast for the rest of the year
        for t in range(len(rest_actual)):
            next_forecast = alpha * rest_actual[t] + (1 - alpha) * forecast[-1]
            forecast.append(next_forecast)


        mse = mean_squared_error(rest_actual, forecast[len(jan_actual):])
        mad = mean_absolute_deviation(rest_actual, forecast[len(jan_actual):])

        if mse < best_mse:
            best_mse = mse
            best_alpha_mse = alpha
            best_forecast_mse = forecast[len(jan_actual):]

        if mad < best_mad:
            best_mad = mad
            best_alpha_mad = alpha
            best_forecast_mad = forecast[len(jan_actual):]

    return best_alpha_mse, best_mse, best_forecast_mse, best_alpha_mad, best_mad, best_forecast_mad

# Holt's Linear Trend Model
def holt_linear_trend(data, alpha, beta, forecast_steps=0):
    # Calculate T0 (slope) using the provided formula
    n = len(data)
    t = np.arange(n)
    sum_t = np.sum(t)
    sum_t_squared = np.sum(t**2)
    sum_data = np.sum(data)
    sum_t_data = np.sum(t * data)

    T0 = (n * sum_t_data - sum_t * sum_data) / (n * sum_t_squared - sum_t**2)

    # Calculate L0 (intercept)
    L0 = (sum_data / n) - (T0 * (sum_t / n))

    # Initialize levels and trends using L0 and T0
    levels = [L0]
    trends = [T0]
    forecasts = []


    for t in range(1, len(data)):

        level = alpha * data[t] + (1 - alpha) * (levels[-1] + trends[-1])


        trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]


        levels.append(level)
        trends.append(trend)


        for m in range(1, forecast_steps + 1):

            forecast = levels[-1] + m * trends[-1]
            forecasts.append(forecast)

    return levels, trends, forecasts

def find_best_params(data):
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]  # Alpha values to test
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]  # Beta values to test

    best_alpha_mse, best_beta_mse, best_error_mse = None, None, float('inf')
    best_alpha_mad, best_beta_mad, best_error_mad = None, None, float('inf')

    for alpha in alphas:  # Loop through each alpha value
        for beta in betas:  # Loop through each beta value
            levels, trends, _ = holt_linear_trend(data, alpha, beta)
            fitted = [levels[t] + trends[t] for t in range(len(levels))]

            # Calculate MSE for current alpha and beta
            mse = mean_squared_error(data[1:], fitted[:-1])

            # Calculate MAD for current alpha and beta
            mad = mean_absolute_deviation(data[1:], fitted[:-1])

            if mse < best_error_mse:  
                best_alpha_mse, best_beta_mse, best_error_mse = alpha, beta, mse

            if mad < best_error_mad:  # Check if current MAD is the best
                best_alpha_mad, best_beta_mad, best_error_mad = alpha, beta, mad

    return (best_alpha_mse, best_beta_mse, best_error_mse,
            best_alpha_mad, best_beta_mad, best_error_mad)


def calculate_daily_seasonal_factors(data):
    data = data.copy()
    data.set_index('datetime', inplace=True)

    # Ensure the time series is complete and forward-fill missing days
    data = data.resample('D').mean().ffill()

    # Check the number of rows and ensure at least two cycles
    required_length = 2 * 365  # 2 years of daily data
    if len(data) < required_length:
        print("Padding data to meet required length for seasonal decomposition.")
        data = pd.concat([data, data], ignore_index=True)
        data = data.iloc[:required_length]

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data['tempmax'], model='multiplicative', period=30)

    # Extract and normalize seasonal factors
    seasonal_factors = decomposition.seasonal
    normalized_factors = seasonal_factors / seasonal_factors.mean()

    # Reset index and return as DataFrame
    factors_with_dates = normalized_factors.reset_index()
    factors_with_dates.columns = ['Day', 'Seasonal_Factor']

    return factors_with_dates

def holts_winter_forecasting(data, alpha, beta, gamma, seasonal_factors):
    tempmax = data['tempmax'].values
    days_of_year = data['datetime'].dt.day_of_year.values
    n = len(tempmax)

    seasonality = seasonal_factors.set_index('Day')['Seasonal_Factor'].to_dict()
    seasonal_factors_mapped = [seasonality.get(day, 1) if seasonality.get(day) is not None else 1.0 for day in days_of_year]

    t = np.arange(n)
    sum_t = np.sum(t)
    sum_t_squared = np.sum(t**2)
    sum_data = np.sum(tempmax)
    sum_t_data = np.sum(t * tempmax)

    T0 = (n * sum_t_data - sum_t * sum_data) / (n * sum_t_squared - sum_t**2)
    L0 = (sum_data / n) - (T0 * (sum_t / n))

    level = [L0]
    trend = [T0]
    seasonality_update = [seasonal_factors_mapped[0]]
    forecast = [(L0 + T0) * seasonality_update[0]]

    for t in range(1, n):
        level.append(alpha * (tempmax[t] / seasonal_factors_mapped[t]) + (1 - alpha) * (level[t - 1] + trend[t - 1]))
        trend.append(beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1])
        seasonality_update.append(gamma * (tempmax[t] / level[t]) + (1 - gamma) * seasonal_factors_mapped[t])
        forecast.append((level[t - 1] + trend[t - 1]) * seasonality_update[t])

    return forecast

def find_best_params_winter_whole(data, seasonal_factors):
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_alpha, best_beta, best_gamma, best_error, best_mad = None, None, None, float('inf'), float('inf')
    best_forecast_mad = None

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                forecasts = holts_winter_forecasting(data, alpha, beta, gamma, seasonal_factors)
                mse = mean_squared_error(data['tempmax'], forecasts)
                mad = mean_absolute_deviation(data['tempmax'], forecasts)

                if mse < best_error:
                    best_alpha, best_beta, best_gamma, best_error = alpha, beta, gamma, mse

                if mad < best_mad:
                    best_mad = mad
                    best_forecast_mad = forecasts

    return best_alpha, best_beta, best_gamma, best_error, best_mad, best_forecast_mad

def main():

    # Program  execution
    file_path = "Athens 2021-01-01 to 2023-01-01.csv"
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Separate the data for 2021 and 2022
    data_2021, data_2022 = separate_yearly_data(data)

    #For 2021
    print("For 2021:")
    df_2021 = pd.DataFrame(data_2021)
    outliers_count = remove_outliers(df_2021, 'tempmax')

    #For 2022
    print("For 2022:")
    df_2022 = pd.DataFrame(data_2022)
    outliers_count = remove_outliers(df_2022, 'tempmax')

    # Calculate the forecasted data for 2021 with 7-Day Moving Average Method
    forecasted_data_7_day_2021 = calculate_7day_moving_average(data_2021, 2021)

    # Calculate the forecasted data for 2021 with 7-Day Weighted Moving Average Method
    forecasted_data_7_day_weighted_2021 = calculate_7day_weighted_moving_average(data_2021, 2021)

    # Extract the relevant columns for 2021 (for MSE calculation)
    dates_2021 = []
    tempmax_values_2021 = []
    moving_average_dates_7_2021 = []
    moving_average_values_7_2021 = []
    moving_average_weighted_values_7_2021 = []

    for row in data_2021:
        dates_2021.append(datetime.strptime(row['datetime'], '%Y-%m-%d'))  # Convert to datetime objects
        tempmax_values_2021.append(float(row['tempmax']))

    for item in forecasted_data_7_day_2021:
        if '2021-02-01' <= item['datetime'] <= '2021-12-31':
            moving_average_dates_7_2021.append(datetime.strptime(item['datetime'], '%Y-%m-%d'))
            moving_average_values_7_2021.append(item['forecasted_tempmax'])

    for item in forecasted_data_7_day_weighted_2021:
        if '2021-02-01' <= item['datetime'] <= '2021-12-31':
            moving_average_weighted_values_7_2021.append(item['forecasted_tempmax'])

    # Convert the forecasted data into DataFrame for 7-Day Moving Average
    forecast_results_MA_7_2021 = pd.DataFrame({
        "Date": moving_average_dates_7_2021,
        "Forecast_Temp": moving_average_values_7_2021
    })

    # Convert the forecasted data into DataFrame for 7-Day Weighted Moving Average
    forecast_results_MA_Weighted_7_2021 = pd.DataFrame({
        "Date": moving_average_dates_7_2021,
        "Forecast_Temp_Weighted": moving_average_weighted_values_7_2021
    })

    # Print the DataFrame for 2021 7-Day Moving Average
    print("\nForecasted Results for 2021 of 7-Day Moving Average Method:")
    print(forecast_results_MA_7_2021)

    # Print the DataFrame for 2021 7-Day Weighted Moving Average
    print("\nForecasted Results for 2021 of 7-Day Weighted Moving Average Method:")
    print(forecast_results_MA_Weighted_7_2021)

    # Save both forecast results to CSV
    forecast_results_MA_7_2021.to_csv("7-DAY_Moving_Average_forecasts_2021.csv", index=False)
    forecast_results_MA_Weighted_7_2021.to_csv("7-DAY_Weighted_Moving_Average_forecasts_2021.csv", index=False)
    print("\nFull forecast saved to '7-DAY_Moving_Average_forecasts_2021.csv' and '7-DAY_Weighted_Moving_Average_forecasts_2021.csv'")

    # Calculate and print MSE for 2021 (7-Day Moving Average)
    mse_2021 = mean_squared_error(tempmax_values_2021[6:], moving_average_values_7_2021)  # Skip first 7 days for forecast
    print(f"\nMean Squared Error for 2021 (7-Day Moving Average): {mse_2021}")

    # Calculate and print MSE for 2021 (7-Day Weighted Moving Average)
    mse_2021_weighted = mean_squared_error(tempmax_values_2021[6:], moving_average_weighted_values_7_2021)  # Skip first 7 days for forecast
    print(f"\nMean Squared Error for 2021 (7-Day Weighted Moving Average): {mse_2021_weighted}")

    # Calculate and print MAD for 2021 (7-Day Moving Average)
    mad_2021 = mean_absolute_deviation(tempmax_values_2021[6:], moving_average_values_7_2021)  # Skip first 7 days for forecast
    print(f"\nMean Absolute Deviation for 2021 (7-Day Moving Average): {mad_2021}")

    # Calculate and print MAD for 2021 (7-Day Weighted Moving Average)
    mad_2021_weighted = mean_absolute_deviation(tempmax_values_2021[6:], moving_average_weighted_values_7_2021)  # Skip first 7 days for forecast
    print(f"\nMean Absolute Deviation for 2021 (7-Day Weighted Moving Average): {mad_2021_weighted}")


    # Calculate the forecasted data for 2022 with 7-Day Moving Average Method
    forecasted_data_7_day_2022 = calculate_7day_moving_average(data_2022, 2022)

    # Calculate the forecasted data for 2022 with 7-Day Weighted Moving Average Method
    forecasted_data_7_day_weighted_2022 = calculate_7day_weighted_moving_average(data_2022, 2022)

    # Extract the relevant columns for 2022 (for MSE calculation)
    dates_2022 = []
    tempmax_values_2022 = []
    moving_average_dates_7_2022 = []
    moving_average_values_7_2022 = []
    moving_average_weighted_values_7_2022 = []

    for row in data_2022:
        dates_2022.append(datetime.strptime(row['datetime'], '%Y-%m-%d'))  # Convert to datetime objects
        tempmax_values_2022.append(float(row['tempmax']))

    for item in forecasted_data_7_day_2022:
        if '2022-02-01' <= item['datetime'] <= '2022-12-31':
            moving_average_dates_7_2022.append(datetime.strptime(item['datetime'], '%Y-%m-%d'))
            moving_average_values_7_2022.append(item['forecasted_tempmax'])

    for item in forecasted_data_7_day_weighted_2022:
        if '2022-02-01' <= item['datetime'] <= '2022-12-31':
            moving_average_weighted_values_7_2022.append(item['forecasted_tempmax'])

    # Convert the forecasted data into DataFrame for 7-Day Moving Average
    forecast_results_MA_7_2022 = pd.DataFrame({
        "Date": moving_average_dates_7_2022,
        "Forecast_Temp": moving_average_values_7_2022
    })

    # Convert the forecasted data into DataFrame for 7-Day Weighted Moving Average
    forecast_results_MA_Weighted_7_2022 = pd.DataFrame({
        "Date": moving_average_dates_7_2022,
        "Forecast_Temp_Weighted": moving_average_weighted_values_7_2022
    })

    # Print the DataFrame for 2022 7-Day Moving Average
    print("\nForecasted Results for 2022 of 7-Day Moving Average Method:")
    print(forecast_results_MA_7_2022)

    # Print the DataFrame for 2022 7-Day Weighted Moving Average
    print("\nForecasted Results for 2022 of 7-Day Weighted Moving Average Method:")
    print(forecast_results_MA_Weighted_7_2022)

    # Save both forecast results to CSV
    forecast_results_MA_7_2022.to_csv("7-DAY_Moving_Average_forecasts_2022.csv", index=False)
    forecast_results_MA_Weighted_7_2022.to_csv("7-DAY_Weighted_Moving_Average_forecasts_2022.csv", index=False)
    print("\nFull forecast saved to '7-DAY_Moving_Average_forecasts_2022.csv' and '7-DAY_Weighted_Moving_Average_forecasts_2022.csv'")

    # Calculate and print MSE for 2022 (7-Day Moving Average)
    mse_2022 = mean_squared_error(tempmax_values_2022[6:], moving_average_values_7_2022)  # Skip first 7 days for forecast
    print(f"\nMean Squared Error for 2022 (7-Day Moving Average): {mse_2022}")

    # Calculate and print MSE for 2021 (7-Day Weighted Moving Average)
    mse_2022_weighted = mean_squared_error(tempmax_values_2022[6:], moving_average_weighted_values_7_2022)  # Skip first 7 days for forecast
    print(f"\nMean Squared Error for 2022 (7-Day Weighted Moving Average): {mse_2022_weighted}")

    # Calculate and print MAD for 2021 (7-Day Moving Average)
    mad_2022 = mean_absolute_deviation(tempmax_values_2022[6:], moving_average_values_7_2022)  # Skip first 7 days for forecast
    print(f"\nMean Absolute Deviation for 2022 (7-Day Moving Average): {mad_2022}")

    # Calculate and print MAD for 2021 (7-Day Weighted Moving Average)
    mad_2022_weighted = mean_absolute_deviation(tempmax_values_2022[6:], moving_average_weighted_values_7_2022)  # Skip first 7 days for forecast
    print(f"\nMean Absolute Feviation for 2022 (7-Day Weighted Moving Average): {mad_2022_weighted}")

    #----------------5DAY- Moving Average ---------
    forecasted_data_5_day_2021 = calculate_5day_moving_average(data_2021, 2021)
    forecasted_data_5_day_weighted_2021 = calculate_5day_weighted_moving_average(data_2021, 2021)

    dates_2021 = []
    tempmax_values_2021 = []
    moving_average_dates_5_2021 = []
    moving_average_values_5_2021 = []
    moving_average_weighted_values_5_2021 = []

    for row in data_2021:
        dates_2021.append(datetime.strptime(row['datetime'], '%Y-%m-%d'))
        tempmax_values_2021.append(float(row['tempmax']))

    for item in forecasted_data_5_day_2021:
        if '2021-02-01' <= item['datetime'] <= '2021-12-31':
            moving_average_dates_5_2021.append(datetime.strptime(item['datetime'], '%Y-%m-%d'))
            moving_average_values_5_2021.append(item['forecasted_tempmax'])

    for item in forecasted_data_5_day_weighted_2021:
        if '2021-02-01' <= item['datetime'] <= '2021-12-31':
            moving_average_weighted_values_5_2021.append(item['forecasted_tempmax'])

    forecast_results_MA_5_2021 = pd.DataFrame({
        "Date": moving_average_dates_5_2021,
        "Forecast_Temp": moving_average_values_5_2021
    })

    forecast_results_MA_Weighted_5_2021 = pd.DataFrame({
        "Date": moving_average_dates_5_2021,
        "Forecast_Temp_Weighted": moving_average_weighted_values_5_2021
    })

    print("\nForecasted Results for 2021 of 5-Day Moving Average Method:")
    print(forecast_results_MA_5_2021)

    print("\nForecasted Results for 2021 of 5-Day Weighted Moving Average Method:")
    print(forecast_results_MA_Weighted_5_2021)

    forecast_results_MA_5_2021.to_csv("5-DAY_Moving_Average_forecasts_2021.csv", index=False)
    forecast_results_MA_Weighted_5_2021.to_csv("5-DAY_Weighted_Moving_Average_forecasts_2021.csv", index=False)

    mse_2021 = mean_squared_error(tempmax_values_2021[4:], moving_average_values_5_2021)
    print(f"\nMean Squared Error for 2021 (5-Day Moving Average): {mse_2021}")

    mse_2021_weighted = mean_squared_error(tempmax_values_2021[4:], moving_average_weighted_values_5_2021)
    print(f"\nMean Squared Error for 2021 (5-Day Weighted Moving Average): {mse_2021_weighted}")

    # Calculate and print MAD for 2021 (5-Day Moving Average)
    mad_2021 = mean_absolute_deviation(tempmax_values_2021[4:], moving_average_values_5_2021)  # Skip first 5 days for forecast
    print(f"\nMean Absolute Deviation for 2021 (5-Day Moving Average): {mad_2021}")

    # Calculate and print MAD for 2021 (5-Day Weighted Moving Average)
    mad_2021_weighted = mean_absolute_deviation(tempmax_values_2021[4:], moving_average_weighted_values_5_2021)  # Skip first 5 days for forecast
    print(f"\nMean Absolute Deviation for 2021 (5-Day Weighted Moving Average): {mad_2021_weighted}")

    forecasted_data_5_day_2022 = calculate_5day_moving_average(data_2022, 2022)
    forecasted_data_5_day_weighted_2022 = calculate_5day_weighted_moving_average(data_2022, 2022)

    dates_2022 = []
    tempmax_values_2022 = []
    moving_average_dates_5_2022 = []
    moving_average_values_5_2022 = []
    moving_average_weighted_values_5_2022 = []

    for row in data_2022:
        dates_2022.append(datetime.strptime(row['datetime'], '%Y-%m-%d'))
        tempmax_values_2022.append(float(row['tempmax']))

    for item in forecasted_data_5_day_2022:
        if '2022-02-01' <= item['datetime'] <= '2022-12-31':
            moving_average_dates_5_2022.append(datetime.strptime(item['datetime'], '%Y-%m-%d'))
            moving_average_values_5_2022.append(item['forecasted_tempmax'])

    for item in forecasted_data_5_day_weighted_2022:
        if '2022-02-01' <= item['datetime'] <= '2022-12-31':
            moving_average_weighted_values_5_2022.append(item['forecasted_tempmax'])

    forecast_results_MA_5_2022 = pd.DataFrame({
        "Date": moving_average_dates_5_2022,
        "Forecast_Temp": moving_average_values_5_2022
    })

    forecast_results_MA_Weighted_5_2022 = pd.DataFrame({
        "Date": moving_average_dates_5_2022,
        "Forecast_Temp_Weighted": moving_average_weighted_values_5_2022
    })

    print("\nForecasted Results for 2022 (February 1 to December 31) of 5-Day Moving Average Method:")
    print(forecast_results_MA_5_2022)

    print("\nForecasted Results for 2022 (February 1 to December 31) of 5-Day Weighted Moving Average Method:")
    print(forecast_results_MA_Weighted_5_2022)

    forecast_results_MA_5_2022.to_csv("5-DAY_Moving_Average_forecasts_2022.csv", index=False)
    forecast_results_MA_Weighted_5_2022.to_csv("5-DAY_Weighted_Moving_Average_forecasts_2022.csv", index=False)

    mse_2022 = mean_squared_error(tempmax_values_2022[4:], moving_average_values_5_2022)
    print(f"\nMean Squared Error for 2022 (5-Day Moving Average): {mse_2022}")

    mse_2022_weighted = mean_squared_error(tempmax_values_2022[4:], moving_average_weighted_values_5_2022)
    print(f"\nMean Squared Error for 2022 (5-Day Weighted Moving Average): {mse_2022_weighted}")

    # Calculate and print MAD for 2022 (5-Day Moving Average)
    mad_2022 = mean_absolute_deviation(tempmax_values_2022[4:], moving_average_values_5_2022)  # Skip first 5 days for forecast
    print(f"\nMean Absolute Deviation for 2021 (5-Day Moving Average): {mad_2021}")

    # Calculate and print MAD for 2022 (5-Day Weighted Moving Average)
    mad_2022_weighted = mean_absolute_deviation(tempmax_values_2022[4:], moving_average_weighted_values_5_2022)  # Skip first 5 days for forecast
    print(f"\nMean Absolute Deviation for 2022 (5-Day Weighted Moving Average): {mad_2022_weighted}")

    # Plot the graph of 5-Day (MA) and Actual Max Temperature for 2021
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2021, tempmax_values_2021, label='Original Max Temperature (°C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_5_2021, moving_average_values_5_2021, label='5-Day Moving Average', color='blue', linestyle='-', marker='x')
    plt.title('Max Temperature and 5-Day Moving Average (°C) in Athens for 2021')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tempmax_and_5day_moving_average2021.png")

    # Plot the graph of 5-Day (MA) and Actual Max Temperature for 2022
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2022, tempmax_values_2022, label='Original Max Temperature (°C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_5_2022, moving_average_values_5_2022, label='5-Day Moving Average', color='blue', linestyle='-', marker='x')
    plt.title('Max Temperature and 5-Day Moving Average (°C) in Athens for 2022')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tempmax_and_5day_moving_average2022.png")

    # Plot the graph of 7-Day (MA) and Actual Max Temperature for 2021
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2021, tempmax_values_2021, label='Original Max Temperature (°C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_7_2021, moving_average_values_7_2021, label='7-Day Moving Average', color='blue', linestyle='-', marker='x')
    plt.title('Max Temperature and 7-Day Moving Average (°C) in Athens for 2021')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tempmax_and_7day_moving_average2021.png")

    # Plot the graph of 7-Day (MA) and Actual Max Temperature for 2022
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2022, tempmax_values_2022, label='Original Max Temperature (°C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_7_2022, moving_average_values_7_2022, label='7-Day Moving Average', color='blue', linestyle='-', marker='x')
    plt.title('Max Temperature and 7-Day Moving Average (°C) in Athens for 2022')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tempmax_and_7day_moving_average2022.png")

    # Plot the graph of Actual Max Temperature for 2021
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2021, tempmax_values_2021, label='Max Temperature (°C)', color='blue')
    plt.title('Max Temperature (°C) in Athens for 2021')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_tempmax2021.png")

    # Plot the graph of Actual Max Temperature for 2022
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2022, tempmax_values_2022, label='Max Temperature (°C)', color='blue')
    plt.title('Max Temperature (°C) in Athens for 2022')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_tempmax2022.png")

# Additional plots for 7-Day and 5-Day Weighted Moving Averages for both years

    # Plot 7-Day Weighted Moving Average for 2021
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2021, tempmax_values_2021, label='Original Max Temperature (\u00b0C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_7_2021, moving_average_weighted_values_7_2021, label='7-Day Weighted Moving Average', color='green', linestyle='-', marker='x')
    plt.title('Max Temperature and 7-Day Weighted Moving Average (\u00b0C) in Athens for 2021')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (\u00b0C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("7day_weighted_moving_average_2021.png")

    # Plot 7-Day Weighted Moving Average for 2022
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2022, tempmax_values_2022, label='Original Max Temperature (\u00b0C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_7_2022, moving_average_weighted_values_7_2022, label='7-Day Weighted Moving Average', color='green', linestyle='-', marker='x')
    plt.title('Max Temperature and 7-Day Weighted Moving Average (\u00b0C) in Athens for 2022')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (\u00b0C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("7day_weighted_moving_average_2022.png")

    # Plot 5-Day Weighted Moving Average for 2021
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2021, tempmax_values_2021, label='Original Max Temperature (\u00b0C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_5_2021, moving_average_weighted_values_5_2021, label='5-Day Weighted Moving Average', color='purple', linestyle='-', marker='x')
    plt.title('Max Temperature and 5-Day Weighted Moving Average (\u00b0C) in Athens for 2021')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (\u00b0C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("5day_weighted_moving_average_2021.png")

    # Plot 5-Day Weighted Moving Average for 2022
    plt.figure(figsize=(12, 6))
    plt.plot(dates_2022, tempmax_values_2022, label='Original Max Temperature (\u00b0C)', color='red', linestyle='-', marker='o')
    plt.plot(moving_average_dates_5_2022, moving_average_weighted_values_5_2022, label='5-Day Weighted Moving Average', color='purple', linestyle='-', marker='x')
    plt.title('Max Temperature and 5-Day Weighted Moving Average (\u00b0C) in Athens for 2022')
    plt.xlabel('Date')
    plt.ylabel('Max Temperature (\u00b0C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("5day_weighted_moving_average_2022.png")


#-----------Exponensial Smoothing---------------------------

    file_path = "Athens 2021-01-01 to 2023-01-01.csv"  # File path for the input data
    data = load_data(file_path)  # Load and preprocess the data

    # Separate data for 2021 and 2022
    data_2021 = data[data['datetime'].dt.year == 2021].copy()
    data_2022 = data[data['datetime'].dt.year == 2022].copy()

    # Filter data for January and the rest of the months
    jan_2021 = data_2021[data_2021['datetime'].dt.month == 1]
    rest_2021 = data_2021[data_2021['datetime'].dt.month > 1]
    jan_2022 = data_2022[data_2022['datetime'].dt.month == 1]
    rest_2022 = data_2022[data_2022['datetime'].dt.month > 1]

    # Get maximum temperatures as lists
    jan_temp_2021 = jan_2021['tempmax'].tolist()
    rest_temp_2021 = rest_2021['tempmax'].tolist()
    jan_temp_2022 = jan_2022['tempmax'].tolist()
    rest_temp_2022 = rest_2022['tempmax'].tolist()

    # Define specific alpha values to test
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Forecast for 2021 based on January data
    print("Finding best alpha for 2021 data...")
    best_alpha_mse_2021, best_mse_2021, forecast_mse_2021, best_alpha_mad_2021, best_mad_2021, forecast_mad_2021 = find_best_alpha(jan_temp_2021, rest_temp_2021, alpha_values)
    print(f"Best alpha for 2021 (MSE): {best_alpha_mse_2021} with MSE: {best_mse_2021}")
    print(f"Best alpha for 2021 (MAD): {best_alpha_mad_2021} with MAD: {best_mad_2021}")

    # Create separate DataFrames for 2021 Forecast (MSE and MAD)
    forecast_df_2021_mse = pd.DataFrame({
         "Date": rest_2021['datetime'].dt.date,  # Convert datetime to just date
         "Actual Temperature": rest_2021['tempmax'].tolist(),
         "Forecasted Temperature (MSE)": forecast_mse_2021
    })

    forecast_df_2021_mad = pd.DataFrame({
         "Date": rest_2021['datetime'].dt.date,  # Convert datetime to just date
         "Actual Temperature": rest_2021['tempmax'].tolist(),
         "Forecasted Temperature (MAD)": forecast_mad_2021
    })

    print("2021 Forecast DataFrame (MSE) with Exponensial Smoothing:")
    print(forecast_df_2021_mse)
    print("2021 Forecast DataFrame (MAD) with Exponensial Smoothing:")
    print(forecast_df_2021_mad)

    # Forecast for 2022 based on January data
    print("Finding best alpha for 2022 data...")
    best_alpha_mse_2022, best_mse_2022, forecast_mse_2022, best_alpha_mad_2022, best_mad_2022, forecast_mad_2022 = find_best_alpha(jan_temp_2022, rest_temp_2022, alpha_values)
    print(f"Best alpha for 2022 with Exponensial Smoothing (MSE): {best_alpha_mse_2022} with MSE: {best_mse_2022}")
    print(f"Best alpha for 2022 with Exponensial Smoothing (MAD): {best_alpha_mad_2022} with MAD: {best_mad_2022}")

    # Create separate DataFrames for 2022 Forecast (MSE and MAD)
    forecast_df_2022_mse = pd.DataFrame({
        "Date": rest_2022['datetime'].dt.date,  # Convert datetime to just date
        "Actual Temperature": rest_2022['tempmax'].tolist(),
        "Forecasted Temperature (MSE)": forecast_mse_2022
    })

    forecast_df_2022_mad = pd.DataFrame({
        "Date": rest_2022['datetime'].dt.date,  # Convert datetime to just date
        "Actual Temperature": rest_2022['tempmax'].tolist(),
        "Forecasted Temperature (MAD)": forecast_mad_2022
    })

    print("2022 Forecast DataFrame (MSE) with Exponensial Smoothing:")
    print(forecast_df_2022_mse)
    print("2022 Forecast DataFrame (MAD) with Exponensial Smoothing:")
    print(forecast_df_2022_mad)

    # Save the forecasts to CSV files
    forecast_df_2021_mse.to_csv("Forecast_2021_MSE_Exponensial_Smoothing.csv", index=False)
    forecast_df_2021_mad.to_csv("Forecast_2021_MAD_Exponensial_Smoothing.csv", index=False)
    forecast_df_2022_mse.to_csv("Forecast_2022_MSE_Exponensial_Smoothing.csv", index=False)
    forecast_df_2022_mad.to_csv("Forecast_2022_MAD_Exponensial_Smoothing.csv", index=False)
    print("Forecasts with Exponensial Smoothing saved to 'Forecast_2021_MSE.csv', 'Forecast_2021_MAD.csv', 'Forecast_2022_MSE.csv', and 'Forecast_2022_MAD.csv'.")

    # Generate and save plots for 2021
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df_2021_mse["Date"], forecast_df_2021_mse["Actual Temperature"], label="Actual Temperature", color='blue', linestyle='-', marker='o')
    plt.plot(forecast_df_2021_mse["Date"], forecast_df_2021_mse["Forecasted Temperature (MSE)"], label="Forecasted Temperature (MSE)", color='orange', linestyle='-', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Exponential Smoothing Forecast for 2021 (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Exponential_Smoothing_2021_MSE.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df_2021_mad["Date"], forecast_df_2021_mad["Actual Temperature"], label="Actual Temperature", color='blue', linestyle='-', marker='o')
    plt.plot(forecast_df_2021_mad["Date"], forecast_df_2021_mad["Forecasted Temperature (MAD)"], label="Forecasted Temperature (MAD)", color='green', linestyle='-', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Exponential Smoothing Forecast for 2021 (MAD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Exponential_Smoothing_2021_MAD.png")
    plt.close()

    # Generate and save plots for 2022
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df_2022_mse["Date"], forecast_df_2022_mse["Actual Temperature"], label="Actual Temperature", color='blue', linestyle='-', marker='o')
    plt.plot(forecast_df_2022_mse["Date"], forecast_df_2022_mse["Forecasted Temperature (MSE)"], label="Forecasted Temperature (MSE)", color='orange', linestyle='-', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Exponential Smoothing Forecast for 2022 (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Exponential_Smoothing_2022_MSE.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(forecast_df_2022_mad["Date"], forecast_df_2022_mad["Actual Temperature"], label="Actual Temperature", color='blue', linestyle='-', marker='o')
    plt.plot(forecast_df_2022_mad["Date"], forecast_df_2022_mad["Forecasted Temperature (MAD)"], label="Forecasted Temperature (MAD)", color='green', linestyle='-', marker='x')
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Exponential Smoothing Forecast for 2022 (MAD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Exponential_Smoothing_2022_MAD.png")
    plt.close()

    print("Plots saved as PNG images of Exponensial Smoothing.")



#------------Holt Linear Trend Method---------------

    data_2021 = data[data['datetime'].dt.year == 2021]  # Extract 2021 data
    data_2022 = data[data['datetime'].dt.year == 2022]  # Extract 2022 data

    # Print out the number of rows for each year to verify the split
    print(f"Data for 2021: {data_2021.shape[0]} rows")
    print(f"Data for 2022: {data_2022.shape[0]} rows")

    actual_2021 = data_2021['tempmax'].tolist()  # Extract temperature values for 2021
    actual_2022 = data_2022['tempmax'].tolist()  # Extract temperature values for 2022

    for year, actual, data_year in zip([2021, 2022], [actual_2021, actual_2022], [data_2021, data_2022]):  # Loop through each year
        print(f"Analyzing {year} data...")  # Indicate the year being analyzed

        best_alpha_mse, best_beta_mse, best_error_mse, \
        best_alpha_mad, best_beta_mad, best_error_mad = find_best_params(actual)  # Find best parameters

        print(f"{year} - Best Alpha (MSE): {best_alpha_mse}, Best Beta (MSE): {best_beta_mse}, Best MSE: {best_error_mse}")
        print(f"{year} - Best Alpha (MAD): {best_alpha_mad}, Best Beta (MAD): {best_beta_mad}, Best MAD: {best_error_mad}")

        levels, trends, forecasts = holt_linear_trend(actual, best_alpha_mse, best_beta_mse)  # Apply Holt's model (MSE)
        forecasts_mse = [levels[t] + trends[t] for t in range(len(levels))]  # Generate MSE forecasts

        levels, trends, forecasts = holt_linear_trend(actual, best_alpha_mad, best_beta_mad)  # Apply Holt's model (MAD)
        forecasts_mad = [levels[t] + trends[t] for t in range(len(levels))]  # Generate MAD forecasts

        forecast_df_mse = pd.DataFrame({  # Create DataFrame for MSE forecasts
            "Date": data_year['datetime'],
            "Actual Temperature": actual,
            "Forecasted Temperature (MSE)": forecasts_mse
        })

        forecast_df_mad = pd.DataFrame({  # Create DataFrame for MAD forecasts
            "Date": data_year['datetime'],
            "Actual Temperature": actual,
            "Forecasted Temperature (MAD)": forecasts_mad
        })

        print(f"{year} Forecast DataFrame (MSE) with Holt's Linear Trend Method:")  # Print MSE DataFrame
        print(forecast_df_mse)

        print(f"{year} Forecast DataFrame (MAD) with Holt's Linear Trend Method:")  # Print MAD DataFrame
        print(forecast_df_mad)

        forecast_df_mse.to_csv(f"Holt_Linear_Forecasts_{year}_MSE.csv", index=False)  # Save MSE DataFrame to CSV
        forecast_df_mad.to_csv(f"Holt_Linear_Forecasts_{year}_MAD.csv", index=False)  # Save MAD DataFrame to CSV

        # Visualize and save the results for the year
        plt.figure(figsize=(12, 6))  # Create a figure for the plot

        # Plot the actual max temperature (original data) as a blue line
        plt.plot(data_year['datetime'], data_year['tempmax'], label="Actual Max Temp (°C)", color="blue", linestyle='-', marker='o')

        # Plot the forecasted values (MSE) as an orange line
        plt.plot(data_year['datetime'], forecasts_mse, label="Forecasted Temp (MSE) (°C)", color="orange", linestyle='-', marker='x')

        # Plot the forecasted values (MAD) as a green line
        plt.plot(data_year['datetime'], forecasts_mad, label="Forecasted Temp (MAD) (°C)", color="green", linestyle='-', marker='x')

        # Add a title to the plot
        plt.title(f"Max Temperature vs Forecasted Max Temperature with Hol't Linear Trend Method for {year}")

        # Label the x and y axes
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

        # Add a grid for better readability
        plt.grid(True)

        # Add a legend to differentiate between actual and forecasted temperatures
        plt.legend()

        # Adjust the layout to ensure nothing is clipped (e.g., axis labels)
        plt.tight_layout()

        # Save the plot to a PNG file
        plt.savefig(f"Holt_Linear_Forecasts_{year}.png")  # Save the plot with the year in the filename


#-------------------Holt Winter----------

    data_2021 = data[data['datetime'].dt.year == 2021]  # Extract 2021 data
    data_2022 = data[data['datetime'].dt.year == 2022]  # Extract 2022 data

    # Initialize an empty list to store forecast results for both years
    all_forecast_results_winter_whole = []

    # Process both years (2021 and 2022)
    for year, data_year in [(2021, data_2021), (2022, data_2022)]:

        if not data_year.empty:
            # Calculate seasonal factors based on the data for the current year
            seasonal_factors = calculate_daily_seasonal_factors(data_year)

            # Find the best parameters (alpha, beta, gamma) for Holt's Winter method
            best_alpha, best_beta, best_gamma, best_error, best_mad, best_forecast_mad = find_best_params_winter_whole(data_year, seasonal_factors)

            print(f"\nBest Smoothing Parameters for {year}:")
            print(f"Alpha (Level): {best_alpha}")
            print(f"Beta (Trend): {best_beta}")
            print(f"Gamma (Seasonality): {best_gamma}")
            print(f"Minimum Mean Squared Error (MSE) for {year}: {best_error}")
            print(f"Minimum Mean Absolute Deviation (MAD) for {year}: {best_mad}")

            # Generate forecasts using Holt's Winter method
            forecasts = holts_winter_forecasting(data_year, best_alpha, best_beta, best_gamma, seasonal_factors)
            data_year['Forecast'] = forecasts

            # Create a DataFrame with the relevant columns for the forecasted results
            forecast_results_year = data_year[['datetime', 'tempmax', 'Forecast']]

            # Print the results for the current year
            print(f"\nDaily Forecasts of Holt's Winter Method for {year}:")
            print(forecast_results_year)

            # Create a second DataFrame for the minimum MAD forecast
            mad_forecast_df = data_year.copy()
            mad_forecast_df['Forecast'] = best_forecast_mad

            print(f"\nForecast with Minimum MAD for {year}:")
            print(mad_forecast_df[['datetime', 'tempmax', 'Forecast']])

            # Append the forecast results to the list
            all_forecast_results_winter_whole.append(forecast_results_year)

            # Plot the results and save them to PNG images
            plt.figure(figsize=(10, 6))
            plt.plot(data_year['datetime'], data_year['tempmax'], label='Actual', color='blue')
            plt.plot(data_year['datetime'], forecasts, label='Forecast', color='orange')
            plt.title(f"Temperature Forecast vs Actual ({year})")
            plt.xlabel('Date')
            plt.ylabel('Temperature')
            plt.legend()
            plt.grid()
            plt.savefig(f"Holt's Winter Forecast_vs_Actual_{year}.png")
            plt.close()

    # Concatenate all forecast results for both years into a single DataFrame
    forecast_df = pd.concat(all_forecast_results_winter_whole, ignore_index=True)

    # Print the concatenated DataFrame
    print("\nComplete Forecast Results of Holt's Winter Method for 2021 and 2022:")
    print(forecast_df)

    # Save the results to a CSV file for both years
    forecast_df.to_csv('Holt_Winter_Forecasts_2021_2022_Whole_Year.csv', index=False)
    print("\nForecast results saved to 'Holt_Winter_Forecasts_2021_2022_Whole_Year.csv'")


    # Combined plot for Holt's Linear Trend MSE and Holt's Winter MSE for 2021
    plt.figure(figsize=(12, 6))
    plt.plot(data_2021['datetime'], data_2021['tempmax'], label='Actual Values (2021)', color='blue')
    plt.plot(data_2021['datetime'], forecast_df_mse['Forecasted Temperature (MSE)'], label="Holt's Linear Trend (MSE)", color='orange', linestyle='-', marker='o')
    plt.plot(data_2021['datetime'], data_2021['Forecast'], label="Holt's Winter Forecast (MSE)", color='red', linestyle='-', marker='x')
    plt.title("Combined Forecast: Holt's Linear Trend MSE & Holt's Winter MSE (2021)")
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid()
    plt.savefig('Combined_Holt_2021.png')
    plt.close()

    # Combined plot for Holt's Linear Trend MSE and Holt's Winter MSE for 2022
    plt.figure(figsize=(12, 6))
    plt.plot(data_2022['datetime'], data_2022['tempmax'], label='Actual Values (2022)', color='blue')
    plt.plot(data_2022['datetime'], forecast_df_mse['Forecasted Temperature (MSE)'], label="Holt's Linear Trend (MSE)", color='orange', linestyle='-', marker='o')
    plt.plot(data_2022['datetime'], data_2022['Forecast'], label="Holt's Winter Forecast (MSE)", color='red', linestyle='-', marker='o')
    plt.title("Combined Forecast: Holt's Linear Trend MSE & Holt's Winter MSE (2022)")
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid()
    plt.savefig('Combined_Holt_2022.png')
    plt.close()

    print("Combined PNG images for Holt's Linear Trend MSE and Holt's Winter MSE forecasts have been saved for 2021 and 2022.")
#------------      Question regarding E --------------------------------
    file_paths = [
        'Alexandroupoli 2021-01-01 to 2022-12-31.csv',
        'Corfu Greece 2021-01-01 to 2022-12-31.csv',
        'Crete Greece 2021-01-01 to 2022-12-31.csv',
        'Xanthi Greece 2021-01-01 to 2022-12-31.csv',
        'kalamata greece 2021-01-01 to 2022-12-31.csv',
        'karpenisi greece 2021-01-01 to 2022-12-31.csv',
        'kavala, greece 2021-01-01 to 2022-12-31.csv',
        'Samos greece 2021-01-01 to 2022-12-31.csv',
        'Thesalloniki 2021-01-01 to 2022-12-31.csv',
        'Zakinthos greece 2021-01-01 to 2022-12-31.csv',
        'Athens 2021-01-01 to 2023-01-01.csv'
    ]

    # Dictionary to store raw values by file
    file = defaultdict(list)

    # Process each file to extract data and perform exponential smoothing
    for file_path in file_paths:
        data_2021 = extract_first_day_data_for_year(file_path)

        if data_2021.empty:
            print(f"No data for year 2021 in {file_path}.")
            continue

        for index, row in data_2021.iterrows():
            value2 = row.get('tempmax', 'N/A')  # Adjust column name if needed
            file[file_path].append(value2)

    # Smoothing factor
    alpha = 0.5
    smoothed_data = {}

    # Apply exponential smoothing for each file
    for key, values in file.items():
        smoothed_data[key] = exponential_smoothing_12(values, alpha)

    # Dictionary to store DataFrames for smoothed data with daily dates
    forecasted_dfs = {}

    # Create DataFrames for each file's smoothed data
    for key, values in smoothed_data.items():
        dates = pd.date_range(start='2021-01-01', periods=len(values), freq='D')  # Adjust start date as necessary
        df = pd.DataFrame({
            'Date': dates,
            'Forecasted (Smoothed) Values': values
        })
        forecasted_dfs[key] = df  # Store DataFrame for each file

    # Get the actual values for Athens
    value1 = file.get('Athens 2021-01-01 to 2023-01-01.csv')
    athens_df = pd.read_csv('Athens 2021-01-01 to 2023-01-01.csv')

    # Ensure 'datetime' column is parsed
    athens_df['datetime'] = pd.to_datetime(athens_df['datetime'], errors='coerce')

    # Filter data for the first day of each month
    athens_first_day = athens_df.resample('MS', on='datetime').first()
    athens_actual_values = athens_first_day['tempmax'].values

    # Calculate MSE for each forecasted DataFrame
    for key, df in forecasted_dfs.items():
        if key != 'Athens 2021-01-01 to 2023-01-01.csv':  # Skip Athens file itself
            forecasted_values = df['Forecasted (Smoothed) Values'].values
            mse_value = mean_squared_error(athens_actual_values[:len(forecasted_values)], forecasted_values)
            print(f"MSE for forecasted data from {key} compared to Athens: {mse_value}")

    Mean_prediction = [15.9 ,16.69, 15.79 , 16.07, 21.56 ,22.61, 29, 33.17,32.42, 27.75, 23.48 , 18.82 ]
    Mean_MSE = mean_squared_error(value1, Mean_prediction)
    print(f"MSE for forecasted data for daily mean: {Mean_MSE}")


# Run the program when the script is executed
if __name__ == "__main__":
    main()
