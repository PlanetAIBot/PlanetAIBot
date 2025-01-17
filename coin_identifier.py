import requests

class CoinIdentifier:
    def __init__(self):
        self.api_url = 'https://api.coingecko.com/api/v3/coins/markets'
    
    def fetch_coin_data(self):
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 10,
            'page': 1
        }
        response = requests.get(self.api_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print("Failed to fetch coin data.")
            return []
