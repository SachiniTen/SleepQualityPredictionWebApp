import unittest
from unittest.mock import patch
import pandas as pd
import joblib
from SleepPrediction import SleepScorePredictor, StreamlitUI, RecommendationEngine  # Assuming 'main.py' is your file

class TestSleepScorePrediction(unittest.TestCase):

    def setUp(self):
        # Load a sample model and scaler for testing
        self.model = joblib.load('models_new/ridge_pipeline.pkl')
        self.scaler = joblib.load('models_new/scaler.pkl')

    @patch('streamlit.sidebar.number_input')
    def test_enter_screen_time_phone(self, mock_number_input):
        # Validate the system's ability to enter daily screen time for phone/tablet
        mock_number_input.return_value = 5.0
        screen_time_phone = StreamlitUI.display_sidebar_inputs()[0]
        self.assertEqual(screen_time_phone, 5.0)

    @patch('streamlit.sidebar.number_input')
    def test_enter_screen_time_laptop(self, mock_number_input):
        # Validate the system's ability to enter daily screen time for laptop/computer
        mock_number_input.return_value = 3.0
        screen_time_laptop = StreamlitUI.display_sidebar_inputs()[1]
        self.assertEqual(screen_time_laptop, 3.0)

    @patch('streamlit.sidebar.selectbox')
    def test_select_screen_content(self, mock_selectbox):
        # Validate the system's ability to select screen content type
        mock_selectbox.return_value = 'Social Media'
        screen_content = StreamlitUI.display_sidebar_inputs()[3]
        self.assertEqual(screen_content, 'Social Media')

    @patch('streamlit.sidebar.number_input')
    def test_enter_step_count(self, mock_number_input):
        # Validate the system's ability to enter daily step count
        mock_number_input.return_value = 8000
        step_count = StreamlitUI.display_sidebar_inputs()[2]
        self.assertEqual(step_count, 8000)

    @patch('streamlit.selectbox')
    def test_select_regression_model(self, mock_selectbox):
        # Validate the system's ability to select a regression model
        mock_selectbox.return_value = 'Random Forest Regression'
        selected_model = StreamlitUI.handle_predictions(SleepScorePredictor(), 5.0, 3.0, 8000, 'Social Media', RecommendationEngine(),
                                       'Random Forest Regression')[6]
        self.assertEqual(selected_model, 'Random Forest Regression')

    def test_predict_sleep_quality_score(self):
        # Validate the system's ability to predict sleep quality score
        screentime = 5.0
        screen_content = 'Social Media'
        step_count = 8000
        predicted_score = SleepScorePredictor.predict_sleep_score(screentime, screen_content, step_count, self.model,
                                                                  self.scaler, 'Random Forest Regression')
        self.assertIsInstance(predicted_score, int)

    @patch('streamlit.write')
    def test_display_user_inputs(self, mock_write):
        # Validate the system's ability to display user inputs after prediction
        screentime = 18.0
        screen_content = 'Social Media'
        step_count = 4000
        predicted_score = SleepScorePredictor.predict_sleep_score(screentime, screen_content, step_count, self.model,
                                                                  self.scaler, 'Linear Regression')
        StreamlitUI.display_recommendations(predicted_score, screentime, screen_content, step_count, self.model,
                                            self.scaler, 'Linear Regression', RecommendationEngine())
        self.assertTrue(mock_write.called)
        calls = [call[0][0] for call in mock_write.call_args_list]
        self.assertIn(f"-User spends an average of {screentime} hours on his phone/tablet/laptop/computer.\n", calls)
        self.assertIn(f"-User watches content belonging to {screen_content} category before bed\n", calls)
        self.assertIn(f"-User has an average daily step count of {step_count}\n", calls)

    def test_display_evaluation_metrics(self):
        # Validate the system's ability to display evaluation metrics
        r2, mae, mse, rmse = SleepScorePredictor.evaluate_model(self.model, 'Random Forest Regression',
                                                                pd.DataFrame([[6.0, 3, 9000]]),
                                                                pd.DataFrame([[6.0, 3, 9000]]), [65])
        self.assertIsInstance(r2, float)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(rmse, float)

    @patch('streamlit.button')
    def test_display_metrics_on_button_click(self, mock_button):
        # Validate metrics are displayed only when the button is clicked
        mock_button.return_value = True
        result = StreamlitUI.evaluation_metrics(SleepScorePredictor, 'Random Forest Regression')
        self.assertIsNotNone(result)

    def test_generate_recommendations_below_60(self):
        # Validate recommendations are generated if sleep score is below 60
        recommendation = RecommendationEngine.generate_recommendations_screentime(55, 6.0, 'Social Media', 9000,
                                                                                  self.model, self.scaler,
                                                                                  'Linear Regression')
        self.assertIn("decrease his screen time", recommendation)

    def test_generate_no_recommendations_above_60(self):
        # Validate no recommendations if sleep score is 60 or above
        recommendation = RecommendationEngine.generate_recommendations_screentime(60, 6.0, 'Social Media', 9000,
                                                                                  self.model, self.scaler,
                                                                                  'Linear Regression')
        self.assertEqual(recommendation,
                         "No recommendations needed as sleep score is in a fair sleep score range (60 or above)")

    def test_display_recommendations_tab(self):
        # Validate the system displays recommendations in a dedicated "Recommendations" tab
        with patch('streamlit.write') as mock_write:
            StreamlitUI.display_recommendations(55, 6.0, 'Social Media', 9000, self.model, self.scaler,
                                                'Linear Regression', RecommendationEngine)
            mock_write.assert_called()

    def test_display_no_prediction_message(self):
        # Validate the system displays a message if no prediction has been made
        with patch('streamlit.write') as mock_write:
            StreamlitUI.display_recommendations(None, 6.0, 'Social,Media', 9000, self.model, self.scaler,
                                                'Linear Regression', RecommendationEngine)
            mock_write.assert_called_with("Please make a prediction in the Prediction tab first.")


if __name__ == '__main__':
    unittest.main()
