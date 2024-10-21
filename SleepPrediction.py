import math
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class SleepScorePredictor:
    @staticmethod
    def load_model(model_file):
        model = joblib.load(model_file)
        return model

    @staticmethod
    def evaluate_model(model, model_nm, X_test, X_test_scaled, y_test):
        if model_nm == 'Linear Regression':
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # Format the metrics to 5 decimal places
        r2 = format(r2, ".5f")
        mae = format(mae, ".5f")
        mse = format(mse, ".5f")
        rmse = format(rmse, ".5f")
        
        return r2, mae, mse, rmse

    @staticmethod
    def predict_sleep_score(screentime, screen_content, step_count, model, scaler, model_nm):
        screen_content_mapping = {
            'Educational Content': 0,
            'News': 1,
            'Relaxing / Meditative Content': 2,
            'Social Media': 3
        }
        screen_content_code = screen_content_mapping[screen_content]

        input_df = pd.DataFrame({
            'Screentime': [screentime],
            'Screen_content_code': [screen_content_code],
            'Step count': [step_count]
        })

        if model_nm == 'Linear Regression':
            sleep_score_pred = model.predict(input_df)
        else:
            input_scaled = scaler.transform(input_df)
            sleep_score_pred = model.predict(input_scaled)

        return math.ceil(sleep_score_pred[0])


class RecommendationEngine:
    @staticmethod
    def generate_recommendations_screentime(predicted_score, current_screentime, screen_content, step_count, model, scaler, model_nm):
        if predicted_score < 60:
            decrement = 1
            max_iterations = 74
            iterations = 0

            while current_screentime > 0 and iterations < max_iterations:
                current_screentime -= decrement
                new_score = SleepScorePredictor.predict_sleep_score(current_screentime, screen_content, step_count, model, scaler, model_nm)

                print("------------start-------------------------")
                print("new_score: ", new_score)
                print("current_screentime", current_screentime)
                print("screen_content", screen_content)
                print("step count", step_count)
                print("model_nm", model_nm)
                print("-------------------------------------")
                print("\n")
                if new_score >= 60:
                    return f"User should decrease his screen time by {iterations + 1} hours in order to improve his sleep quality to the fair sleep score range (above or equal to 60)."
                iterations += 1

            return "Unable to reach the target sleep score with reasonable screen time reduction."
        else:
            return "No recommendations needed as sleep score is in a fair sleep score range (60 or above)"

    @staticmethod
    def generate_recommendations_stepcount(predicted_score, screentime, screen_content, current_step_count, model, scaler, model_nm):
        if predicted_score < 60:
            increment = 100
            max_iterations1 = 10000
            iterations = 0

            while current_step_count > 0 and iterations < max_iterations1 and current_step_count < 11000:
                current_step_count += increment
                new_score_1 = SleepScorePredictor.predict_sleep_score(screentime, screen_content, current_step_count, model, scaler, model_nm)

                print("------------start 2-------------------------")
                print("new_score: ", new_score_1)
                print("screentime", screentime)
                print("screen_content", screen_content)
                print("current step count", current_step_count)
                print("model_nm", model_nm)
                print("-------------------------------------")
                print("\n")
                if new_score_1 >= 60:
                    return f"User should increase his step count by {iterations + 100} steps in order to improve his sleep quality to the fair sleep score range (above or equal to 60)."
                iterations += 100

            return "Unable to reach the target sleep score with a reasonable step count increase."
        else:
            return "No recommendations needed as sleep score is in a fair sleep score range (60 or above)"

    @staticmethod
    def generate_recommendations_screencontent(predicted_score, screentime, current_screen_content, step_count, model, scaler, model_nm):
        if predicted_score < 60:
            content_types = ["News", "Social Media", "Educational Content", "Relaxing / Meditative Content"]
            recommendations = []

            for content in content_types:
                if content != current_screen_content:
                    new_score = SleepScorePredictor.predict_sleep_score(screentime, content, step_count, model, scaler, model_nm)

                    print("------------start-------------------------")
                    print("new_score: ", new_score)
                    print("current_screentime", screentime)
                    print("screen_content", content)
                    print("step count", step_count)
                    print("model_nm", model_nm)
                    print("-------------------------------------")
                    print("\n")

                    recommendations.append(f"Switching to {content} content may change the sleep score to {new_score}.\n")

            if recommendations:
                return "\n".join(recommendations)
            else:
                return "Unable to reach the target sleep score with reasonable screen content changes."

        else:
            return "No recommendations needed as sleep score is in a fair sleep score range (60 or above)."


class StreamlitUI:
    @staticmethod
    def display_title():
        st.title('💤 Sleep Quality Prediction Web App')

    @staticmethod
    def display_sidebar_inputs():
        st.sidebar.title('Please enter the following: ')
        screen_time_phone = st.sidebar.number_input('Average Daily Screen time of phone/tablet (hours)', min_value=0.0, max_value=24.0, value=6.0, step=0.1)
        screen_time_laptop = st.sidebar.number_input('Average Daily Screen time of laptop/computer (hours)', min_value=0.0, max_value=24.0, value=6.0, step=0.1)
        step_count_1 = st.sidebar.number_input('Average Daily Step count', min_value=0, max_value=60000, value=9000, step=100)
        screen_content_1 = st.sidebar.selectbox('Screen content (what you watched within 2 hours before sleeping)',
                                                ['Relaxing / Meditative Content', 'Educational Content', 'Social Media', 'News'])
        return screen_time_phone, screen_time_laptop, step_count_1, screen_content_1

    @staticmethod
    def display_tabs():
        return st.tabs(["Prediction", "Recommendations"])

    @staticmethod
    def handle_predictions(sleep_app, screen_time_phone, screen_time_laptop, step_count_1, screen_content_1, recommendation_engine, model_selection):
        total_screen_time = screen_time_phone + screen_time_laptop
        step_count = step_count_1
        screen_content = screen_content_1



        model_files = {
            'Linear Regression': 'models_new/ridge_pipeline.pkl',
            'Random Forest Regression': 'models_new/best_random_model.pkl',
            'Decision Tree Regression': 'models_new/best_decision_model.pkl',
            'KNN Regression': 'models_new/best_knn_model.pkl',
            'SVM Regression': 'models_new/best_svm_regressor.pkl'
        }

        model_file = model_files[model_selection]
        model = sleep_app.load_model(model_file)
        scaler_loaded = joblib.load('models_new/scaler.pkl')
        #X_test = joblib.load('models_new/X_test.pkl')
        #X_test_scaled = joblib.load('models_new/X_test_scaled.pkl')
        #y_test = joblib.load('models_new/y_test.pkl')

        predicted_sleep_score = sleep_app.predict_sleep_score(total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection)

        #if st.sidebar.button('Predict Sleep Score'):
        if predicted_sleep_score < 0:
            st.write("Invalid input parameters provided by user. Please re-enter the input values.")
        else:
            st.write(f"**Predicted Sleep Score (0-100):** {predicted_sleep_score}")

        #if st.button('View Model Evaluation Metrics'):
        #    r2, mae, mse, rmse = sleep_app.evaluate_model(model, model_selection, X_test, X_test_scaled, y_test)
        #    st.write("### Evaluation Metrics")
        #    st.write(f"**R2 Score:** {r2}")
        #    st.write(f"**Mean Absolute Error:** {mae}")
        #    st.write(f"**Mean Squared Error:** {mse}")
        #    st.write(f"**Root Mean Squared Error:** {rmse}")

        return predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection

    @staticmethod
    def evaluation_metrics(sleep_app,model_selection):
        model_files = {
            'Linear Regression': 'models_new/ridge_pipeline.pkl',
            'Random Forest Regression': 'models_new/best_random_model.pkl',
            'Decision Tree Regression': 'models_new/best_decision_model.pkl',
            'KNN Regression': 'models_new/best_knn_model.pkl',
            'SVM Regression': 'models_new/best_svm_regressor.pkl'
        }

        model_file = model_files[model_selection]
        model = sleep_app.load_model(model_file)
        X_test = joblib.load('models_new/X_test.pkl')
        X_test_scaled = joblib.load('models_new/X_test_scaled.pkl')
        y_test = joblib.load('models_new/y_test.pkl')

        r2, mae, mse, rmse = sleep_app.evaluate_model(model, model_selection, X_test, X_test_scaled, y_test)
        #st.write("### Evaluation Metrics")
        #st.write(f"**R2 Score:** {r2}")
        #st.write(f"**Mean Absolute Error:** {mae}")
        #st.write(f"**Mean Squared Error:** {mse}")
        #st.write(f"**Root Mean Squared Error:** {rmse}")

        return r2, mae, mse, rmse

    @staticmethod
    def display_recommendations(predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection, recommendation_engine):
        if predicted_sleep_score is None:
            st.write("Please make a prediction in the Prediction tab first.")
            return
        if predicted_sleep_score < 0:
            st.write("Invalid input parameters provided by user. Please re-enter the input values.")
        elif predicted_sleep_score < 60:


            if total_screen_time > 0:
                # Predicted sleep score in red font
                st.markdown(f"<span style='color:red;'>**Predicted sleep score: {predicted_sleep_score}.\nThis is below the fair sleep quality threshold of 60.\n**</span>",
                            unsafe_allow_html=True)


                st.write("**User Inputs:**")
                st.write(
                    f"-User spends an average of {total_screen_time} hours on his phone/tablet/laptop/computer.\n")
                st.write(f"-User watches content belonging to {screen_content} category before bed\n")
                st.write(f"-User has an average daily step count of {step_count}\n")

                st.write("**Recommendations:**")
                recommendation = recommendation_engine.generate_recommendations_screentime(predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection)
                st.write(f"{recommendation}\n")

                recommendation_2 = recommendation_engine.generate_recommendations_stepcount(predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection)
                st.write(f"{recommendation_2}\n")

                recommendation_3 = recommendation_engine.generate_recommendations_screencontent(predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection)
                st.write(f"{recommendation_3}\n")
            #else:
            #    st.write( "No recommendations needed as sleep score is in a fair sleep score range (60 or above).")
        elif predicted_sleep_score >= 60:
            st.write("No recommendations needed as sleep score is in a fair sleep score range (60 or above).")




def main():
    ui = StreamlitUI()
    sleep_score_predictor = SleepScorePredictor()
    recommendation_engine = RecommendationEngine()

    ui.display_title()
    screen_time_phone, screen_time_laptop, step_count_1, screen_content_1 = ui.display_sidebar_inputs()
    tabs = ui.display_tabs()

    with tabs[0]:
        model_selection = st.selectbox('Select Regression Model',
                                     ['Random Forest Regression', 'Linear Regression', 'Decision Tree Regression',
                                        'KNN Regression', 'SVM Regression'], index=0, key='model_select',
                                       format_func=lambda x: ' ' * 5 + x)
        st.markdown("""
                <style>
                .stButton > button {
                    background-color: #000053;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)

        if st.sidebar.button('Predict Sleep Quality Score'):
            predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection = \
                ui.handle_predictions(sleep_score_predictor, screen_time_phone, screen_time_laptop, step_count_1, screen_content_1, recommendation_engine,model_selection)
        if st.button('View Model Evaluation Metrics'):
            r2, mae, mse, rmse = ui.evaluation_metrics(sleep_score_predictor, model_selection)
            st.write("### Evaluation Metrics")
            st.write(f"**R2 Score:** {r2}")
            st.write(f"**Mean Absolute Error:** {mae}")
            st.write(f"**Mean Squared Error:** {mse}")
            st.write(f"**Root Mean Squared Error:** {rmse}")
    with tabs[1]:
        if 'predicted_sleep_score' in locals():
            ui.display_recommendations(predicted_sleep_score, total_screen_time, screen_content, step_count, model, scaler_loaded, model_selection, recommendation_engine)

        else:
            st.write("Please make a prediction in the Prediction tab first.")
    # Dark theme and improved styling
    # Dark theme and improved styling
    # Define your custom CSS
    custom_css = """
    <style>
    .sidebar .sidebar-content {
        background-color: #251c52;
        color: #262730;
    }
    </style>
    """

    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
