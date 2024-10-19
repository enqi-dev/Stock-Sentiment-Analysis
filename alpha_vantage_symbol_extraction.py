import pandas as pd

API_KEY = 'demo'  # Your AV API key

ACTIVE_CSV_URL = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=demo"

symbols = pd.read_csv(ACTIVE_CSV_URL)

print(symbols[['symbol', 'name']])

symbols.to_csv("stock_symbols.csv")
