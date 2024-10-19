import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import streamlit as st

logos = {'AAPL': 'apple',
         'AMZN': 'amazon',
         'GRAB': 'grab',
         'NOK': 'nokia',
         'SE': 'shopee',
         'SONY': 'sony',
         'T': 'at&t',
         'TM': 'toyota',
         'TSLA': 'tesla',
         'UBER': 'uber'
         }


def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]  # Return ADF Statistic and p-value


def remove_outliers_percentile(df, column, lower_percentile=0.05, upper_percentile=0.95):
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered


# Streamlit layout
st.title('Stock Price and Sentiment Analysis')

# File upload options
STOCK_FILE = st.file_uploader("Upload Stock Data (Excel)", type=["xlsx"])
SENTIMENT_FILE = st.file_uploader(
    "Upload Sentiment Data (Excel)", type=["xlsx"])

if STOCK_FILE and SENTIMENT_FILE:
    # Load both files
    sentiment_data = pd.ExcelFile(SENTIMENT_FILE)
    stock_data = pd.ExcelFile(STOCK_FILE)

    # Display available sheet names for selection
    sentiment_sheet = st.selectbox(
        "Select Sentiment Data Sheet", sentiment_data.sheet_names)
    stock_sheet = st.selectbox(
        "Select Stock Data Sheet", stock_data.sheet_names)

    # Load the selected sheets into DataFrames
    sentiment_df = pd.read_excel(
        SENTIMENT_FILE, sheet_name=sentiment_sheet)  # Sentiment data
    stock_df = pd.read_excel(STOCK_FILE, sheet_name=stock_sheet)  # Stock data

    # Convert date columns to datetime
    sentiment_df['Date'] = pd.to_datetime(
        sentiment_df['Date'], format='%d/%m/%Y')
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%d/%m/%Y')

    # Select stock for analysis
    STOCK = st.selectbox("Select Stock", stock_df['Stock'].unique())

    # Select stock price metric (Open, Close, High, Low, etc.)
    PRICE = st.selectbox("Select Price Metric", [
                         'Open', 'Close', 'High', 'Low'])

    # Filter stock data based on selected stock
    stock_df = stock_df[['Stock', 'Date', PRICE]]
    stock_df = stock_df[stock_df['Stock'] == STOCK]

    # Filter sentiment data based on the company name
    SENTIMENT_COMPANY = logos[STOCK]
    sentiment_df = sentiment_df.loc[sentiment_df["Company"]
                                    == SENTIMENT_COMPANY]

    # Filter both datasets for a specific year
    YEAR = st.number_input("Enter the Year for Analysis", value=2024)
    stock_df = stock_df[stock_df['Date'].dt.year == YEAR]
    sentiment_df = sentiment_df[sentiment_df['Date'].dt.year == YEAR]

    # Check if there is sufficient data for the selected year
    if stock_df.shape[0] == 0:
        st.error(f"No stock data available for {YEAR}.")
    elif sentiment_df.shape[0] == 0:
        st.error(f"No sentiment data available for {YEAR}.")
    else:
        stock_df = stock_df.dropna()
        stock_df = stock_df.sort_values(by='Date')
        stock_df = stock_df.groupby(
            stock_df['Date'].dt.date).agg({PRICE: 'mean'})
        stock_df = stock_df.reset_index()
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df = stock_df.set_index('Date')

        # Merge stock and sentiment data on the Date column
        merged_df = pd.merge(stock_df, sentiment_df, on='Date')
        merged_df = merged_df.drop(columns=["Time"], errors='ignore')

        # Plot Stock Prices
        plt.figure(figsize=(10, 6))
        plt.plot(stock_df[PRICE], label=f'{PRICE} Prices ({YEAR})')
        plt.title(f'{STOCK} ({SENTIMENT_COMPANY}) - {PRICE} Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel(PRICE)
        plt.legend()
        st.pyplot(plt)

        # Perform the ADF test (stock price)
        adf_stat, p_value = adf_test(stock_df[PRICE])
        st.write(f'ADF Statistic (Stock Price): {adf_stat}')
        st.write(f'p-value (Stock Price): {p_value}')
        if p_value <= 0.05:
            st.success("The stock price series is stationary")
        else:
            st.warning("The stock price series is not stationary")

        # Create a new ARIMA model with the sentiment feature
        model = ARIMA(merged_df[PRICE],
                      exog=merged_df['Compound'], order=(1, 1, 1))
        model_fit = model.fit()

        # Use predict to get fitted values starting from the first non-NaN value
        start = 1  # Can change to match the first valid prediction
        end = len(stock_df)  # Till the end of the dataset
        stock_df['fitted'] = model_fit.predict(
            start=start, end=end, dynamic=False)

        # merged_df = merged_df.merge(merged_df, model_fit.fittedvalues)
        fitted_df = pd.DataFrame(model_fit.fittedvalues, columns=['fitted'])
        stock_df = pd.concat([merged_df, fitted_df], axis=1)

        stock_df = remove_outliers_percentile(stock_df, 'fitted')
        stock_df['fitted'] = model_fit.fittedvalues
        stock_df.set_index('Date', inplace=True)

        # df.to_csv("Test3.csv", index=False)
        # Plot fitted values
        plt.figure(figsize=(10, 6))
        plt.plot(stock_df['fitted'], label='Fitted', color='red')
        plt.title(f'ARIMA Model - Fitted {PRICE} Prices ({sentiment_sheet})')
        plt.xlabel('Date')
        plt.ylabel('Fitted Values')
        plt.legend()
        st.pyplot(plt)

        # Forecast one step ahead using future sentiment value
        future_sentiment = st.number_input(
            "Enter future sentiment value:", value=0.166907)
        forecasted_value = model_fit.forecast(steps=1, exog=[future_sentiment])

        if not forecasted_value.empty:
            # Print the forecasted value
            st.write(f"Forecasted {PRICE} Value: {forecasted_value.values[0]}")
        else:
            st.error("Forecasting failed, please check your input data.")

        # Actual vs Fitted values
        plt.figure(figsize=(10, 6))
        plt.plot(stock_df[PRICE], label='Actual Price', color='blue')
        plt.plot(stock_df['fitted'], label='Fitted', color='red')
        plt.title(f'ARIMA Model - Actual vs. Fitted {PRICE} Prices')
        plt.xlabel('Date')
        plt.legend()
        st.pyplot(plt)
