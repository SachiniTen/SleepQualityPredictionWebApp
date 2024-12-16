# Sleep Quality Prediction Web Application

This Streamlit web application predicts the sleep quality score (1-100) for youth aged 13-30 based on their:
- Average daily screen time (phone, tablet, laptop, or computer)
- Type of screen content viewed before bed
- Average daily step count

If the predicted sleep score falls below 60 (indicating poor sleep quality), the application generates personalized recommendations to improve sleep quality.

Web app: https://sleepqualitypredictionwebapp.streamlit.app/

Source code: https://github.com/SachiniTen/SleepQualityPredictionWebApp/

---


## Features
- **Input Form**: Users can input screen time, screen content, and daily step count.
- **Prediction**: Predicts the sleep quality score using machine learning regression models.
- **Personalized Recommendations**: Provides actionable tips when the predicted score is below 60.
- **Evaluation Metrics**: Displays model evaluation metrics like R², MAE, MSE, and RMSE.


## Installation Guide
Follow these steps to get the application up and running:

### 1. Clone the Repository
```bash
git clone https://github.com/SachiniTen/SleepQualityPredictionWebApp.git
cd SleepQualityPredictionWebApp
```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
```bash
python -m venv myenv
source myenv/bin/activate   # On macOS/Linux
myenv\Scripts\activate    # On Windows
```

### 3. Install Required Libraries
The required libraries and their versions are specified in the `requirements.txt` file. Install them using:
```bash
pip install -r requirements.txt
```

**Included Libraries:**
- **pandas** (v2.2.2): For data manipulation and analysis
- **numpy** (v1.26.4): For numerical computations
- **joblib** (v1.4.2): For saving and loading machine learning models
- **scikit-learn** (v1.5.0): For implementing and evaluating machine learning models
- **streamlit** (v1.35.0): For building the web application

### 4. Run the Application
Start the Streamlit server and launch the application:
```bash
streamlit run SleepPrediction.py
```
This will open the web application in your default web browser.



## Environment Configurations

### Development Tools Used:
1. **PyCharm**: Utilized for front-end development, backend integration, and loading the machine learning models.
2. **Jupyter Notebook**: Used for data preprocessing, exploratory data analysis, model training, and evaluation.


## File Structure
```plaintext
├── SleepPrediction.py    # Main application file
├── requirements.txt      # List of required libraries
├── models/               # Directory containing trained machine learning models
└── README.md             # Project documentation
```

---

## Usage Instructions
1. Input your daily screen time , type of content watched, and step count into the form.
2. Click on 'Predict Sleep Quality Score' toView the predicted sleep quality score.
4. If the score is below 60, click on the 'View Recommendations' button to view the personalized recommendations provided.


---

## Troubleshooting
- **Dependencies Not Installing**: Ensure you are using Python 3.9 or higher and activate your virtual environment before running `pip install`.
- **Streamlit App Not Launching**: Verify that all libraries are installed and the file path to `SleepPrediction.py` is correct.


---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments
Special thanks to the open-source community for providing the tools and resources used in this project.
