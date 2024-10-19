import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import mplfinance as mpf
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the combined dataset containing all stock predictions
data = pd.read_csv('ARIMA_Sentiment_Predictions.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Streamlit app
st.title('Stock Price Forecasting with SARIMA Model')

# User input for stock selection
stock_options = data['Stock'].unique()
selected_stock = st.selectbox('Select Stock', stock_options)

# Filter data based on selected stock
filtered_data = data[data['Stock'] == selected_stock]

# User input for platform selection
platform_options = filtered_data['Platform'].unique()
selected_platform = st.selectbox('Select Platform', platform_options)

# Filter data based on selected platform
filtered_data = filtered_data[filtered_data['Platform'] == selected_platform]

# Convert 'Date' column to year format for filtering
filtered_data['Year'] = filtered_data['Date'].dt.year

# Allow user to filter the year range using a slider
min_year = filtered_data['Year'].min()
max_year = filtered_data['Year'].max()
selected_year_range = st.slider('Select Year Range', min_year, max_year, (min_year, max_year))

# Filter both the actual and predicted data based on the selected year range
filtered_data = filtered_data[
    (filtered_data['Year'] >= selected_year_range[0]) &
    (filtered_data['Year'] <= selected_year_range[1])
    ]

# Separate actual data and predicted data after filtering by date range
actual_data = filtered_data[filtered_data['Open'].notna()]  # Rows with actual data
predicted_data = filtered_data[['Date', 'Predicted_Close']]  # Rows with predicted data


# Ensure both actual and predicted data are not empty
if actual_data.empty or predicted_data.empty:
    st.write("Not enough data available to perform analysis.")
else:
    # Plotting the Actual vs. Predicted Prices
    st.subheader(f'Actual vs. Predicted Stock Prices for {selected_stock}')
    plt.figure(figsize=(14, 7))

    # Plot actual stock prices up to the last available date within the filtered range
    plt.plot(actual_data['Date'], actual_data['Close'], label='Actual Prices', color='blue', alpha=0.7)

    # Plot predicted stock prices within the selected year range
    plt.plot(predicted_data['Date'], predicted_data['Predicted_Close'],
             label='SARIMA Predicted Prices', color='red', alpha=0.7)

    # Formatting the date on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate(rotation=45)

    # Add title and labels
    plt.title(f'SARIMA Model: Actual vs Predicted Stock Prices for {selected_stock}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)

    # --- Histogram of Sentiment Scores ---
    if 'Compound' in filtered_data.columns and not filtered_data.empty:
        st.subheader('Histogram of Sentiment Scores')

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Create the histogram
        plt.figure(figsize=(14, 7))
        sns.histplot(filtered_data['Compound'], bins=30, kde=True,
                     color='purple')  # 'Compound' is the sentiment score column

        # Add titles and labels
        plt.title('Histogram of Sentiment Scores', fontsize=16)
        plt.xlabel('Sentiment Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        # Display the plot in Streamlit
        st.pyplot(plt)
    else:
        st.write("No sentiment score data available for histogram.")

    # --- Area Chart for Cumulative Sentiment ---
    if 'Compound' in filtered_data.columns and not filtered_data.empty:
        st.subheader('Cumulative Sentiment Over Time')

        # Calculate cumulative sentiment
        filtered_data['Cumulative_Sentiment'] = filtered_data['Compound'].cumsum()

        # Create area chart
        plt.figure(figsize=(14, 7))
        plt.fill_between(filtered_data['Date'], filtered_data['Cumulative_Sentiment'], color='purple', alpha=0.5)
        plt.plot(filtered_data['Date'], filtered_data['Cumulative_Sentiment'], color='purple')

        # Add titles and labels
        plt.title('Cumulative Sentiment Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Sentiment Score', fontsize=14)
        plt.xticks(rotation=45)

        # Display the plot in Streamlit
        st.pyplot(plt)
    else:
        st.write("No sentiment score data available for cumulative sentiment.")

    # --- Sentiment Time Series Plot ---

    if 'Compound' in filtered_data.columns:
        st.subheader(f'Sentiment Time Series for {selected_stock}')
        plt.figure(figsize=(14, 7))

        # Plot sentiment over time
        plt.plot(filtered_data['Date'], filtered_data['Compound'], label='Sentiment', color='purple', alpha=0.7)

        # Formatting the date on the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate(rotation=45)

        # Add title and labels
        plt.title(f'Sentiment Over Time for {selected_stock}')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Display the sentiment plot in Streamlit
        st.pyplot(plt)
    else:
        st.write("Sentiment data is not available for the selected stock.")


    # --- Bollinger Bands Overlay ---
    if 'Close' in actual_data.columns:
        st.subheader('Bollinger Bands Overlay')

        # Calculate moving averages and standard deviation
        window = 20
        actual_data['MA20'] = actual_data['Close'].rolling(window=window).mean()
        actual_data['Upper_Band'] = actual_data['MA20'] + (actual_data['Close'].rolling(window=window).std() * 2)
        actual_data['Lower_Band'] = actual_data['MA20'] - (actual_data['Close'].rolling(window=window).std() * 2)

        plt.figure(figsize=(14, 7))
        plt.plot(actual_data['Date'], actual_data['Close'], label='Close Price', color='blue', alpha=0.5)
        plt.plot(actual_data['Date'], actual_data['MA20'], label='20-Day MA', color='red')
        plt.fill_between(actual_data['Date'], actual_data['Upper_Band'], actual_data['Lower_Band'], color='grey',
                         alpha=0.2, label='Bollinger Bands')

        plt.title(f'Bollinger Bands for {selected_stock}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.legend()

        # Display the Bollinger Bands plot
        st.pyplot(plt)


    # --- Residual Plot ---
    # Merge actual data and predicted data on the Date column
    aligned_data = pd.merge(actual_data[['Date', 'Close']], predicted_data, on='Date', how='inner')

    rmse = np.sqrt(mean_squared_error(aligned_data['Close'], aligned_data['Predicted_Close']))
    mae = mean_absolute_error(aligned_data['Close'], aligned_data['Predicted_Close'])

    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")

    st.subheader(f'Residual Plot for {selected_stock}')

    # Check if the merge resulted in any data
    if aligned_data.empty:
        st.write("No aligned data available for residual analysis.")
    else:
        # Calculate residuals (actual - predicted)
        aligned_data['Residuals'] = aligned_data['Close'] - aligned_data['Predicted_Close']
        aligned_data = aligned_data[abs(aligned_data['Residuals']) <= 20]

        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(aligned_data['Date'], aligned_data['Residuals'], color='red', s=10)
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f'Residuals Over Time for {selected_stock}')
        plt.xlabel('Date')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Display the residual plot in Streamlit
        st.pyplot(plt)

    # --- RSI and MACD Calculation ---
    # Calculate RSI
    window_length = 14
    delta = actual_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    actual_data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    short_ema = actual_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = actual_data['Close'].ewm(span=26, adjust=False).mean()
    actual_data['MACD'] = short_ema - long_ema
    actual_data['Signal'] = actual_data['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Moving Averages
    actual_data['MA20'] = actual_data['Close'].rolling(window=20).mean()  # Short-term MA
    actual_data['MA50'] = actual_data['Close'].rolling(window=50).mean()  # Long-term MA

    # Time Series Plot for Indicators
    st.subheader('Technical Indicators Over Time')

    # Create a figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    # Plot RSI
    axs[0].plot(actual_data['Date'], actual_data['RSI'], label='RSI', color='blue')
    axs[0].axhline(70, color='red', linestyle='--', label='Overbought (70)')
    axs[0].axhline(30, color='green', linestyle='--', label='Oversold (30)')
    axs[0].set_title('Relative Strength Index (RSI)')
    axs[0].set_ylabel('RSI Value')
    axs[0].legend()
    axs[0].grid()

    # Plot MACD
    axs[1].plot(actual_data['Date'], actual_data['MACD'], label='MACD', color='purple')
    axs[1].plot(actual_data['Date'], actual_data['Signal'], label='Signal Line', color='orange')
    axs[1].set_title('MACD Indicator')
    axs[1].set_ylabel('MACD Value')
    axs[1].legend()
    axs[1].grid()

    # Plot Moving Averages
    axs[2].plot(actual_data['Date'], actual_data['Close'], label='Close Price', color='blue', alpha=0.5)
    axs[2].plot(actual_data['Date'], actual_data['MA20'], label='20-Day MA', color='red')
    axs[2].plot(actual_data['Date'], actual_data['MA50'], label='50-Day MA', color='green')
    axs[2].set_title('Moving Averages')
    axs[2].set_ylabel('Price')
    axs[2].legend()
    axs[2].grid()

    # Formatting the x-axis
    plt.xticks(rotation=45)
    plt.xlabel('Date')

    # Adjust layout
    plt.tight_layout()

    # Display the time series plots in Streamlit
    st.pyplot(fig)

