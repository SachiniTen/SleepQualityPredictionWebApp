# Sleep Quality Prediction Web Application

This Streamlit web application predicts the sleep quality score (1-100) for youth aged 13-30 based on their:
- Average daily screen time (phone, tablet, laptop, or computer)
- Type of screen content viewed before bed
- Average daily step count

If the predicted sleep score falls below 60 (indicating poor sleep quality), the application generates personalized recommendations to improve sleep quality.

Web app: https://sleepqualitypredictionwebapp.streamlit.app/

Source code: https://github.com/SachiniTen/SleepQualityPredictionWebApp/


Python code used for data preprocessing, exploratory data analysis(EDA), model training, hyperparameter tuning and model evaluation can be found in 2313071.Code.zip -> Sleep_Quality_Prediction_Model.ipynb

Python code of the web application can be found in 
2313071.Code.zip -> SleepQualityPredictionWebApp

---


## Features
- **Input Form**: Users can input screen time, screen content, and daily step count.
- **Prediction**: Predicts the sleep quality score using machine learning regression models.
- **Personalized Recommendations**: Provides actionable recommendation when the predicted score is below 60.

## Prerequisites
Ensure python is installed on your system

## Installation Guide
Follow these steps to get the application up and running:

### 1. Clone the Repository / Open the SleepPredictionWebApp project folder
```bash
git clone https://github.com/SachiniTen/SleepQualityPredictionWebApp.git
cd SleepQualityPredictionWebApp
```
SleepPredictionWebApp project folder can be found in 2313071.Code.zip -> SleepQualityPredictionWebApp

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


### Required External Libraries:
If libraries fail to install via `requirements.txt`, install them manually:
- **pandas**: [Documentation and Installation Guide](https://pandas.pydata.org/)
- **numpy**: [Documentation and Installation Guide](https://numpy.org/)
- **joblib**: [Documentation and Installation Guide](https://joblib.readthedocs.io/)
- **scikit-learn**: [Documentation and Installation Guide](https://scikit-learn.org/stable/install.html)
- **streamlit**: [Documentation and Installation Guide](https://docs.streamlit.io/)

---

## File Structure
```plaintext
├── SleepPrediction.py     # Main application file
├── requirements.txt       # List of required libraries
├── models_final/          # Directory containing trained machine learning models
├── README.md              # Project documentation
├── unit_tests.py          # unit test code
├── integration_testing.py # integration test code
└── locustfile.py          # performance testing code 

```


---

## Usage Instructions
1. Input your daily screen time , type of content watched, and step count into the form.
2. Click on 'Predict Sleep Quality Score' toView the predicted sleep quality score.
4. If the score is below 60, click on the 'View Recommendations' button to view the personalized recommendations provided.


---

## Acknowledgments
Special thanks to the open-source community for providing the tools and resources used in this project.
