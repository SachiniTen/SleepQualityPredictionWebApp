import unittest
from unittest.mock import patch
from SleepPrediction import SleepScorePredictor, StreamlitUI, RecommendationEngine  # Assuming 'SleepPrediction' is your file
import joblib

class TestSleepScorePredictionIntegration(unittest.TestCase):
    def setUp(self):
        # Load a sample model and scaler for testing
        self.model = joblib.load('models_final/best_knn_model.pkl')
        self.scaler = joblib.load('models_final/scaler.pkl')
        self.sleep_score_predictor = SleepScorePredictor()
        self.recommendation_engine = RecommendationEngine()

    @patch('streamlit.sidebar.number_input')
    @patch('streamlit.sidebar.selectbox')
    def test_integration_prediction(self, mock_selectbox, mock_number_input):
        # Set mock values for sidebar inputs
        mock_number_input.side_effect = [6.0, 6.0, 9000]  # phone screen time, laptop screen time, step count
        mock_selectbox.return_value = 'Relaxing / Meditative Content'
        # Simulate user inputs from the UI
        screen_time_phone, screen_time_laptop, step_count_1, screen_content_1 = StreamlitUI.display_sidebar_inputs()
        # Validate that UI collects the correct inputs
        self.assertEqual(screen_time_phone, 6.0)
        self.assertEqual(screen_time_laptop, 6.0)
        self.assertEqual(step_count_1, 9000)
        self.assertEqual(screen_content_1, 'Relaxing / Meditative Content')
        # Perform sleep score prediction
        total_screen_time = screen_time_phone + screen_time_laptop
        predicted_sleep_score = self.sleep_score_predictor.predict_sleep_score(
            total_screen_time, screen_content_1, step_count_1, self.model, self.scaler, 'KNN Regression'
        )
        # Validate that the prediction is successfully made
        self.assertIsInstance(predicted_sleep_score, int)
        # Generate and validate recommendations
        recommendation = self.generate_recommendations(predicted_sleep_score, total_screen_time, screen_content_1, step_count_1)
        if predicted_sleep_score < 60:
            self.assertIn("decrease his screen time", recommendation)
        else:
            self.assertIn("No recommendations", recommendation)

    def generate_recommendations(self, predicted_sleep_score, total_screen_time, screen_content, step_count):
        # Generate recommendations based on sleep score, screen time, screen content, and step count.
        recommendation = self.recommendation_engine.generate_recommendations_screentime(
            predicted_sleep_score, total_screen_time, screen_content, step_count, self.model, self.scaler,
            'KNN Regression'
        )
        return recommendation

if __name__ == '__main__':
    unittest.main()
