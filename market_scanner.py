import time
import random

class MarketScanner:
    def __init__(self, scan_interval=3600, trend_threshold=0.75):
        self.scan_interval = scan_interval
        self.trend_threshold = trend_threshold
    
    def start_scan(self):
        while True:
            print("Scanning for crypto trends...")
            trend = self.scan_for_trends()
            if trend > self.trend_threshold:
                print(f"Trend detected: {trend}")
            time.sleep(self.scan_interval)
    
    def scan_for_trends(self):
        # Simulates scanning for crypto trends
        return random.random()  # Random trend value between 0 and 1
