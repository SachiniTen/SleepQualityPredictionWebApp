import unittest
from unittest.mock import patch
from SleepPrediction import SleepScorePredictor, StreamlitUI, \
    RecommendationEngine  # Assuming 'SleepPrediction' is your file
import joblib


class TestSleepScorePredictionIntegration(unittest.TestCase):

    def setUp(self):
        # Load a sample model and scaler for testing
        self.model = joblib.load('models_new/ridge_pipeline.pkl')
        self.scaler = joblib.load('models_new/scaler.pkl')
        self.sleep_score_predictor = SleepScorePredictor()
        self.recommendation_engine = RecommendationEngine()

    @patch('streamlit.sidebar.number_input')
    @patch('streamlit.sidebar.selectbox')
    def test_integration_prediction_and_recommendation(self, mock_selectbox, mock_number_input):
        # Set mock values for sidebar inputs
        mock_number_input.side_effect = [5.0, 3.0, 8000]  # phone screen time, laptop screen time, step count
        mock_selectbox.return_value = 'Social Media'
         # Simulate user inputs from the UI
        screen_time_phone, screen_time_laptop, step_count_1, screen_content_1 = StreamlitUI.display_sidebar_inputs()

        # Validate that UI collects the correct inputs
        self.assertEqual(screen_time_phone, 5.0)
        self.assertEqual(screen_time_laptop, 3.0)
        self.assertEqual(step_count_1, 8000)
        self.assertEqual(screen_content_1, 'Social Media')
         # Perform sleep score prediction
        total_screen_time = screen_time_phone + screen_time_laptop
        predicted_sleep_score = self.sleep_score_predictor.predict_sleep_score(
            total_screen_time, screen_content_1, step_count_1, self.model, self.scaler, 'Linear Regression'
        )
        # Validate that the prediction is successfully made
        self.assertIsInstance(predicted_sleep_score, int)
         # Validate the integration with the recommendation engine
        recommendation = self.recommendation_engine.generate_recommendations_screentime(
            predicted_sleep_score, total_screen_time, screen_content_1, step_count_1, self.model, self.scaler,
            'Linear Regression'
        )
        # Verify recommendation logic based on predicted sleep score
        if predicted_sleep_score < 60:
            self.assertIn("decrease his screen time", recommendation)
        else:
            self.assertIn("No recommendations", recommendation)

    @patch('streamlit.sidebar.number_input')
    @patch('streamlit.sidebar.selectbox')
    def test_integration_evaluation_metrics(self, mock_selectbox, mock_number_input):
        # Set mock values for sidebar inputs
        mock_number_input.side_effect = [5.0, 3.0, 8000]  # phone screen time, laptop screen time, step count
        mock_selectbox.return_value = 'Linear Regression'

        # Simulate user inputs from the UI
        screen_time_phone, screen_time_laptop, step_count_1, screen_content_1 = StreamlitUI.display_sidebar_inputs()

        # Ensure evaluation metrics can be correctly calculated
        r2, mae, mse, rmse = StreamlitUI.evaluation_metrics(self.sleep_score_predictor, 'Linear Regression')

        # Validate that the metrics return proper float values
        self.assertIsInstance(r2, float)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(rmse, float)


if __name__ == '__main__':
    unittest.main()