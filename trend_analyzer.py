class TrendAnalyzer:
    def __init__(self):
        pass
    
    def identify_trending_coins(self, coins_data):
        trending_coins = []
        for coin in coins_data:
            if coin['price_change_percentage_24h'] > 5:
                trending_coins.append({
                    'name': coin['name'],
                    'potential': coin['price_change_percentage_24h']
                })
        return trending_coins
