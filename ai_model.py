import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class PlanetAIBot:
    def __init__(self):
        # Initialize the AI model and scaler
        self.model = None
        self.scaler = StandardScaler()

    def train(self, market_data, labels):
        """
        Train the AI model using historical market data.
        
        :param market_data: np.ndarray, features of the market (e.g., price, volume, sentiment scores).
        :param labels: np.ndarray, binary labels (e.g., 1 for promising, 0 for not promising).
        """
        # Scale the input data
        scaled_data = self.scaler.fit_transform(market_data)
        
        # Define the neural network
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=scaled_data.shape[1]),
            tf.keras.layers.Dropout(0.3),  # Add dropout for regularization
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        self.model.fit(scaled_data, labels, epochs=15, batch_size=32, verbose=1)

    def predict(self, market_data):
        """
        Predict the potential of coins using trained AI model.
        
        :param market_data: np.ndarray, features of the coins to evaluate.
        :return: np.ndarray, predictions (probability of being promising).
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet. Train the model before making predictions.")
        
        # Scale the input data
        scaled_data = self.scaler.transform(market_data)
        
        # Predict probabilities
        predictions = self.model.predict(scaled_data)
        return predictions

# Example usage
if __name__ == "__main__":
    # Simulated market data (e.g., price, volume, sentiment scores)
    market_data = np.random.rand(100, 5)  # 100 coins with 5 features each
    labels = np.random.randint(0, 2, size=(100,))  # Binary labels (1: promising, 0: not promising)
    
    bot = PlanetAIBot()
    bot.train(market_data, labels)
    
    # New data to predict
    new_data = np.random.rand(10, 5)  # 10 coins to evaluate
    predictions = bot.predict(new_data)
    print("Predictions:", predictions)
