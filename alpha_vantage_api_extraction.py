import requests

API_KEY = 'demo'
BASE_URL = 'https://www.alphavantage.co/query'
symbol = "GRAB"  # changed accordingly to stock symbol


params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': symbol,
    'outputsize': 'full',
    'apikey': API_KEY,
    'datatype': 'csv'

}

response = requests.get(BASE_URL, params=params)
if response.status_code == 200:
    # Save the content to a file
    with open(f'{symbol}_file.csv', 'wb') as file:
        file.write(response.content)
    print('CSV file downloaded and saved successfully.')
else:
    print(f'Failed to download the file. Status code: {response.status_code}')
