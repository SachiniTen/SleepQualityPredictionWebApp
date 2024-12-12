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
    def generate_recommendations_screentime(predicted_score, current_screentime, screen_content, step_count, model,
                                            scaler, model_nm):
        if predicted_score < 60:
            decrement = 1
            max_iterations = 74
            iterations = 0

            while current_screentime > 0 and iterations < max_iterations:
                current_screentime -= decrement
                new_score = SleepScorePredictor.predict_sleep_score(current_screentime, screen_content, step_count,
                                                                    model, scaler, model_nm)

                print("------------start-------------------------")
                print("new_score: ", new_score)
                print("current_screentime", current_screentime)
                print("screen_content", screen_content)
                print("step count", step_count)
                print("model_nm", model_nm)
                print("-------------------------------------")
                print("\n")
                if new_score >= 60:
                    return f"💡User should decrease his screen time by {iterations + 1} hours in order to improve his sleep quality to the fair sleep score range (above or equal to 60)."
                iterations += 1

            return "Unable to reach the target sleep score with reasonable screen time reduction."
        else:
            return "No recommendations needed as sleep score is in a fair sleep score range (60 or above)"

    @staticmethod
    def generate_recommendations_stepcount(predicted_score, screentime, screen_content, current_step_count, model,
                                           scaler, model_nm):
        if predicted_score < 60:
            increment = 100
            max_iterations1 = 10000
            iterations = 0

            while current_step_count > 0 and iterations < max_iterations1 and current_step_count < 110000:
                current_step_count += increment
                new_score_1 = SleepScorePredictor.predict_sleep_score(screentime, screen_content, current_step_count,
                                                                      model, scaler, model_nm)

                print("------------start 2-------------------------")
                print("new_score: ", new_score_1)
                print("screentime", screentime)
                print("screen_content", screen_content)
                print("current step count", current_step_count)
                print("model_nm", model_nm)
                print("-------------------------------------")
                print("\n")
                if new_score_1 >= 60:
                    return f"💡User should increase his step count by {iterations + 100} steps in order to improve his sleep quality to the fair sleep score range (above or equal to 60)."
                iterations += 100

            return "Unable to reach the target sleep score with a reasonable step count increase."
        else:
            return "No recommendations needed as sleep score is in a fair sleep score range (60 or above)"

    @staticmethod
    def generate_recommendations_screencontent(predicted_score, screentime, current_screen_content, step_count, model,
                                               scaler, model_nm):
        if predicted_score < 60:
            content_types = ["News", "Social Media", "Educational Content", "Relaxing / Meditative Content"]
            recommendations = []

            for content in content_types:
                if content != current_screen_content:
                    new_score = SleepScorePredictor.predict_sleep_score(screentime, content, step_count, model, scaler,
                                                                        model_nm)

                    print("------------start-------------------------")
                    print("new_score: ", new_score)
                    print("current_screentime", screentime)
                    print("screen_content", content)
                    print("step count", step_count)
                    print("model_nm", model_nm)
                    print("-------------------------------------")
                    print("\n")

                    recommendations.append(
                        f"💡Switching to {content} content may change the sleep score to {new_score}.\n")

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
        # Predicted sleep score in red font
        st.markdown(
            f"<span style='color:black;'><b><h2>💤 Sleep Quality Prediction Web App</h2></b></span>",
            unsafe_allow_html=True)
    @staticmethod
    def display_sidebar_inputs():
        st.markdown(
            f"<span style='color:black;'><b>Please enter the following: </b></span>",
            unsafe_allow_html=True)
        screen_time_phone = st.number_input('Average Daily Screen time of phone/tablet (hours)', min_value=0.0,
                                                    max_value=24.0, value=6.0, step=0.1)
        screen_time_laptop = st.number_input('Average Daily Screen time of laptop/computer (hours)',
                                                     min_value=0.0, max_value=24.0, value=6.0, step=0.1)
        step_count_1 = st.number_input('Average Daily Step count', min_value=0, max_value=60000, value=9000,
                                               step=100)
        screen_content_1 = st.selectbox('Screen content (what you watched within 2 hours before sleeping)',
                                                ['Relaxing / Meditative Content', 'Educational Content', 'Social Media',
                                                 'News'])
        return screen_time_phone, screen_time_laptop, step_count_1, screen_content_1

    @staticmethod
    def handle_predictions(sleep_app, screen_time_phone, screen_time_laptop, step_count, screen_content,
                           recommendation_engine, model_selection):
        total_screen_time = screen_time_phone + screen_time_laptop
        model_files = {
            'KNN Regression': 'models_new/best_knn_model.pkl'
        }
        model_file = model_files[model_selection]
        model = sleep_app.load_model(model_file)
        scaler = joblib.load('models_new/scaler.pkl')
        predicted_score = sleep_app.predict_sleep_score(total_screen_time, screen_content, step_count, model, scaler,
                                                        model_selection)
        st.session_state.predicted_sleep_score = predicted_score
        st.session_state.prediction_details = (
        total_screen_time, screen_content, step_count, model, scaler, model_selection)
        st.info(f"Predicted Sleep Score: {predicted_score}")
        return predicted_score

    @staticmethod
    def display_recommendations(recommendation_engine):
        if 'predicted_sleep_score' not in st.session_state:
            st.sidebar.error("Please predict the sleep quality score first.")
            st.stop()

        predicted_score = st.session_state.predicted_sleep_score
        total_screen_time, screen_content, step_count, model, scaler, model_selection = st.session_state.prediction_details

        if predicted_score < 60:
            #st.sidebar.write("**Recommendations:**")
            recommendation = recommendation_engine.generate_recommendations_screentime(predicted_score,
                                                                                       total_screen_time,
                                                                                       screen_content, step_count,
                                                                                       model, scaler, model_selection)
            st.sidebar.write(recommendation)

            recommendation_2 = recommendation_engine.generate_recommendations_stepcount(predicted_score,
                                                                                        total_screen_time,
                                                                                        screen_content, step_count,
                                                                                        model, scaler, model_selection)
            st.sidebar.write(recommendation_2)

            recommendation_3 = recommendation_engine.generate_recommendations_screencontent(predicted_score,
                                                                                            total_screen_time,
                                                                                            screen_content, step_count,
                                                                                            model, scaler,
                                                                                            model_selection)
            st.sidebar.write(recommendation_3)
        else:
            st.sidebar.success("No recommendations needed as the sleep score is fair or above.")


def main():
    st.set_page_config(page_title="Sleep Quality Prediction Web App", page_icon="💤", layout="wide")
    st.header("💤 Sleep Quality Prediction Web App")

    ui = StreamlitUI()
    sleep_score_predictor = SleepScorePredictor()
    recommendation_engine = RecommendationEngine()
    # Add a horizontal line

    #ui.display_title()

    st.sidebar.markdown(
        f"<span style='color:black;'><b><h4>💡Recommendations</h4></b></span>",
        unsafe_allow_html=True)

    st.markdown(
        f"<span style='color:black;'>This web application predicts the sleep quality score of youth aged(13-30) based on their avergae daily screen time(phone/tablet), average daily screen time(laptop/computer), screen content(what they watched before going to bed) and average daily step count</span>",
        unsafe_allow_html=True)
    screen_time_phone, screen_time_laptop, step_count_1, screen_content_1 = ui.display_sidebar_inputs()





    model_selection = 'KNN Regression'
    st.markdown("""
            <style>
            .stButton > button {
                background-color: #15317E;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)

    if st.button('Predict Sleep Quality Score'):
        ui.handle_predictions(sleep_score_predictor, screen_time_phone, screen_time_laptop, step_count_1, screen_content_1, recommendation_engine, 'KNN Regression')



    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #D8E5F7;
        }
    </style>
    """, unsafe_allow_html=True)




    if st.sidebar.button('View Recommendations'):
        #ui.handle_predictions(sleep_score_predictor, screen_time_phone, screen_time_laptop, step_count_1, screen_content_1, recommendation_engine,
        #                      'KNN Regression')

        ui.display_recommendations(recommendation_engine)
        ui.handle_predictions(sleep_score_predictor, screen_time_phone, screen_time_laptop, step_count_1,
                              screen_content_1, recommendation_engine,
                                                 'KNN Regression')











    # Define your custom CSS
    custom_css = """
    <style>
    .sidebar .sidebar-content {
        background-color: #D8E5F7;
        color: #262730;
    }
    </style>
    """

    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
