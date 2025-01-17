import pandas as pd

class DataProcessor:
    def __init__(self):
        pass
    
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def clean_data(self, data):
        # Example data cleaning (can be extended)
        data = data.dropna()
        data = data[data['value'] > 0]  # Removes rows where value <= 0
        return data
