import unittest
from unittest.mock import patch
import pandas as pd
import streamlit as st
import joblib
from SleepPrediction import SleepScorePredictor, StreamlitUI, RecommendationEngine  # Assuming 'SleepPrediction.py' is your file



class TestSleepScorePrediction(unittest.TestCase):

    def setUp(self):
        # Load a sample model and scaler for testing
        self.model = joblib.load('models_final/best_knn_model.pkl')
        self.scaler = joblib.load('models_final/scaler.pkl')
        self.mock_session_state = {}

    @patch('streamlit.sidebar.number_input')
    def test_enter_screen_time_phone(self, mock_number_input):
        # Validate the system's ability to enter daily screen time for phone/tablet
        mock_number_input.return_value = 6.0
        screen_time_phone = StreamlitUI.display_sidebar_inputs()[0]
        self.assertEqual(screen_time_phone, 6.0)

    @patch('streamlit.sidebar.number_input')
    def test_enter_screen_time_laptop(self, mock_number_input):
        # Validate the system's ability to enter daily screen time for laptop/computer
        mock_number_input.return_value = 6.0
        screen_time_laptop = StreamlitUI.display_sidebar_inputs()[1]
        self.assertEqual(screen_time_laptop, 6.0)

    @patch('streamlit.sidebar.selectbox')
    def test_select_screen_content(self, mock_selectbox):
        # Validate the system's ability to select screen content type
        mock_selectbox.return_value = 'Relaxing / Meditative Content'
        screen_content = StreamlitUI.display_sidebar_inputs()[3]
        self.assertEqual(screen_content, 'Relaxing / Meditative Content')

    @patch('streamlit.sidebar.number_input')
    def test_enter_step_count(self, mock_number_input):
        # Validate the system's ability to enter daily step count
        mock_number_input.return_value = 9000
        step_count = StreamlitUI.display_sidebar_inputs()[2]
        self.assertEqual(step_count, 9000)



    def test_predict_sleep_quality_score(self):
        # Validate the system's ability to predict sleep quality score
        screentime = 12.0
        screen_content = 'Relaxing / Meditative Content'
        step_count = 9000
        predicted_score = SleepScorePredictor.predict_sleep_score(screentime, screen_content, step_count, self.model,
                                                                  self.scaler, 'KNN Regression')
        self.assertIsInstance(predicted_score, int)



    def test_generate_recommendations_below_60(self):
        # Validate recommendations are generated if sleep score is below 60
        recommendation = RecommendationEngine.generate_recommendations_screentime(55, 6.0, 'Relaxing / Meditative Content', 9000,
                                                                                  self.model, self.scaler,
                                                                                  'KNN Regression')
        self.assertIn("decrease the screen time", recommendation)

    def test_generate_no_recommendations_above_60(self):
        # Validate no recommendations if sleep score is 60 or above
        recommendation = RecommendationEngine.generate_recommendations_screentime(60, 6.0, 'Relaxing / Meditative Content', 9000,
                                                                                  self.model, self.scaler,
                                                                                  'KNN Regression')
        self.assertEqual(recommendation,
                         "No recommendations needed as the sleep quality score is within the fair sleep score range")


    def test_display_no_prediction_message(self):
       self. mock_session_state.clear()
       # Validate the system displays a message if no prediction has been made
       with patch('streamlit.sidebar.error') as mock_write:
            StreamlitUI.display_recommendations(RecommendationEngine)
            mock_write.assert_called_with("Please predict the sleep quality score first.")


if __name__ == '__main__':
    unittest.main()
