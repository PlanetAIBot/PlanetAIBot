import unittest
from src.ai_model.model import AIModel

class TestAIModel(unittest.TestCase):
    
    def test_model_prediction(self):
        model = AIModel()
        # Example data for testing
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        model.train(data)
        prediction = model.predict([[1, 2, 3]])
        self.assertTrue(prediction[0] > 0)

if __name__ == "__main__":
    unittest.main()
