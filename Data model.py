#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary libraries
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split


# In[2]:


# read fundamentals.csv data file
df_main = pd.read_csv('data/dataset_1.csv')
df = df_main.copy()

# # Data Preprocessing

# In[3]:


df.describe()

# In[4]:


df.head()

# In[5]:


# Display the data type of each variable
print("Data Types:\n")
dtypes = df.dtypes
print(dtypes)

# In[6]:


# Initialize empty lists to hold the data types of each variable
quantitative_vars = []
qualitative_vars = []
# Loop through the columns and identify data types of each variable
for col in dtypes.index:
    if dtypes[col] == 'object':
        qualitative_vars.append(col)
    else:
        quantitative_vars.append(col)
# Print the quantitative and qualitative variables
# print all the quantitative variables
# print('Quantitative variables:')
# print(quantitative_vars)
# print the second quantitative variable
print('Quantitative variable:')
print(quantitative_vars)
print('\nQualitative variables:')
print(qualitative_vars)

# In[7]:


# Loop through columns and show available values for categorical variables
for col in df.columns:
    if df[col].dtype == 'object':
        print('\nAvailable Values for {}:'.format(col))
        print(df[col].unique())

# In[8]:


# Get the number of columns
num_columns = len(df.columns)

# Get the number of rows
num_rows = len(df)

print("Number of columns:", num_columns)
print("Number of rows:", num_rows)

# In[9]:


# Drop rows where Screencontent is 'other'
df = df[df['Screencontent'] != 'Other']

# Print the resulting DataFrame
print(df)

# Encoding Categorigal Variables(Screen content)

# In[10]:


# Changing dtypes to categorical variables
df['Screen_content_code'] = df['Screencontent'].astype('category').cat.codes

# In[11]:


df.head()

# In[12]:


df['Screen_content_code'].unique()

# # Model Training

# In[13]:


X = df.loc[:, ['Screentime',
               'Screen_content_code',
               'Step count']]

# In[14]:


X.head(5)

# In[15]:


y = df['Sleep score']

# In[16]:


y.head(5)

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# X_train


# **1.Linear Regression**

# In[18]:


# fitting the general Linear Regression Model and evaluating the accuracy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
regressor_prediction = linear_model.predict(X_test)
linear_R2 = r2_score(y_test, regressor_prediction)

# In[19]:


# Calculate R2 squared
linear_R2 = r2_score(y_test, regressor_prediction)

# Calculate Mean Absolute Error (MAE)
linear_mae = mean_absolute_error(y_test, regressor_prediction)

# Calculate Mean Squared Error (MSE)
linear_mse = mean_squared_error(y_test, regressor_prediction)

# Calculate Root Mean Squared Error (RMSE)
linear_rmse = mean_squared_error(y_test, regressor_prediction, squared=False)

# Print all metrics
print("R2 Score:", linear_R2)
print("Mean Absolute Error (MAE):", linear_mae)
print("Mean Squared Error (MSE):", linear_mse)
print("Root Mean Squared Error (RMSE):", linear_rmse)

# Improve accuracy

# In[20]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Create a pipeline with Polynomial Features, StandardScaler, and Ridge Regression
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

# Fitting the model
pipeline.fit(X_train, y_train)
pipeline_prediction = pipeline.predict(X_test)

# Evaluating the model
pipeline_R2 = r2_score(y_test, pipeline_prediction)
pipeline_MAE = mean_absolute_error(y_test, pipeline_prediction)
pipeline_MSE = mean_squared_error(y_test, pipeline_prediction)
pipeline_RMSE = mean_squared_error(y_test, pipeline_prediction, squared=False)

print(f"Pipeline R2 Score: {pipeline_R2}")
print(f"Pipeline Mean Absolute Error: {pipeline_MAE}")
print(f"Pipeline Mean Squared Error: {pipeline_MSE}")
print(f"Pipeline Root Mean Squared Error: {pipeline_RMSE}")

# **Random Forest**

# In[21]:


from sklearn.ensemble import RandomForestRegressor

random_model = RandomForestRegressor()
random_model.fit(X_train, y_train)
randomforest_prediction = random_model.predict(X_test)

# In[22]:


# Calculate R2 squared
random_R2 = r2_score(y_test, randomforest_prediction)

# Calculate Mean Absolute Error (MAE)
random_mae = mean_absolute_error(y_test, randomforest_prediction)

# Calculate Mean Squared Error (MSE)
random_mse = mean_squared_error(y_test, randomforest_prediction)

# Calculate Root Mean Squared Error (RMSE)
random_rmse = mean_squared_error(y_test, randomforest_prediction, squared=False)

# Print all metrics
print("R2 Score:", random_R2)
print("Mean Absolute Error (MAE):", random_mae)
print("Mean Squared Error (MSE):", random_mse)
print("Root Mean Squared Error (RMSE):", random_rmse)

# In[23]:


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Optionally standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
random_model_1 = RandomForestRegressor(random_state=42)

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=random_model_1, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
best_random_model = grid_search.best_estimator_

# Make predictions
randomforest_prediction_1 = best_random_model.predict(X_test_scaled)

# Evaluate the model
randomforest_R2 = r2_score(y_test, randomforest_prediction_1)
randomforest_MAE = mean_absolute_error(y_test, randomforest_prediction_1)
randomforest_MSE = mean_squared_error(y_test, randomforest_prediction_1)
randomforest_RMSE = mean_squared_error(y_test, randomforest_prediction_1, squared=False)

print(f"Random Forest R2 Score: {randomforest_R2}")
print(f"Random Forest Mean Absolute Error: {randomforest_MAE}")
print(f"Random Forest Mean Squared Error: {randomforest_MSE}")
print(f"Random Forest Root Mean Squared Error: {randomforest_RMSE}")

# 3.Decision Tree Regression Model

# In[24]:


from sklearn.tree import DecisionTreeRegressor

decision_model = DecisionTreeRegressor(max_depth=5)
decision_model.fit(X_train, y_train)
decision_prediction = decision_model.predict(X_test)

decision_R2 = r2_score(y_test, decision_prediction)

# Calculate Mean Absolute Error (MAE)
decision_mae = mean_absolute_error(y_test, decision_prediction)

# Calculate Mean Squared Error (MSE)
decision_mse = mean_squared_error(y_test, decision_prediction)

# Calculate Root Mean Squared Error (RMSE)
decision_rmse = mean_squared_error(y_test, decision_prediction, squared=False)

# Print all metrics
print("R2 Score:", decision_R2)
print("Mean Absolute Error (MAE):", decision_mae)
print("Mean Squared Error (MSE):", decision_mse)
print("Root Mean Squared Error (RMSE):", decision_rmse)

# In[25]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Optionally standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
decision_model_1 = DecisionTreeRegressor(random_state=42)

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=decision_model_1, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
best_decision_model = grid_search.best_estimator_

# Make predictions
decision_prediction_1 = best_decision_model.predict(X_test_scaled)

# Evaluate the model
decision_R2_1 = r2_score(y_test, decision_prediction_1)
decision_MAE_1 = mean_absolute_error(y_test, decision_prediction_1)
decision_MSE_1 = mean_squared_error(y_test, decision_prediction_1)
decision_RMSE_1 = mean_squared_error(y_test, decision_prediction_1, squared=False)

print(f"Decision Tree R2 Score: {decision_R2_1}")
print(f"Decision Tree Mean Absolute Error: {decision_MAE_1}")
print(f"Decision Tree Mean Squared Error: {decision_MSE_1}")
print(f"Decision Tree Root Mean Squared Error: {decision_RMSE_1}")

# KNN Regression

# In[26]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=8)
knn_model.fit(X_train, y_train)
knn_prediction = knn_model.predict(X_test)
knn_R2 = r2_score(y_test, knn_prediction)

# Calculate Mean Absolute Error (MAE)
knn_mae = mean_absolute_error(y_test, knn_prediction)

# Calculate Mean Squared Error (MSE)
knn_mse = mean_squared_error(y_test, knn_prediction)

# Calculate Root Mean Squared Error (RMSE)
knn_rmse = mean_squared_error(y_test, knn_prediction, squared=False)

# Print all metrics
print("R2 Score:", knn_R2)
print("Mean Absolute Error (MAE):", knn_mae)
print("Mean Squared Error (MSE):", knn_mse)
print("Root Mean Squared Error (RMSE):", knn_rmse)

# In[27]:


import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the KNN Regressor model
knn_model_1 = KNeighborsRegressor()

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=knn_model_1, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
best_knn_model = grid_search.best_estimator_

# Make predictions
knn_prediction_1 = best_knn_model.predict(X_test_scaled)

# Evaluate the model
knn_R2_1 = r2_score(y_test, knn_prediction_1)
knn_MAE_1 = mean_absolute_error(y_test, knn_prediction_1)
knn_MSE_1 = mean_squared_error(y_test, knn_prediction_1)
knn_RMSE_1 = mean_squared_error(y_test, knn_prediction_1, squared=False)

print(f"KNN R2 Score: {knn_R2_1}")
print(f"KNN Mean Absolute Error: {knn_MAE_1}")
print(f"KNN Mean Squared Error: {knn_MSE_1}")
print(f"KNN Root Mean Squared Error: {knn_RMSE_1}")

# SVM Regression

# In[28]:


from sklearn.svm import SVR

svm_regressor = SVR(kernel='rbf')
svm_regressor.fit(X_train, y_train)
svm_predict = svm_regressor.predict(X_test)

svm_R2 = r2_score(y_test, svm_predict)

# Calculate Mean Absolute Error (MAE)
svm_mae = mean_absolute_error(y_test, svm_predict)

# Calculate Mean Squared Error (MSE)
svm_mse = mean_squared_error(y_test, svm_predict)

# Calculate Root Mean Squared Error (RMSE)
svm_rmse = mean_squared_error(y_test, svm_predict, squared=False)

# Print all metrics
print("R2 Score:", svm_R2)
print("Mean Absolute Error (MAE):", svm_mae)
print("Mean Squared Error (MSE):", svm_mse)
print("Root Mean Squared Error (RMSE):", svm_rmse)

# In[29]:


import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the SVR model
svm_regressor_1 = SVR()

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=svm_regressor_1, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='r2')

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
best_svm_regressor = grid_search.best_estimator_

# Make predictions
svm_predict_1 = best_svm_regressor.predict(X_test_scaled)

# Evaluate the model
svm_R2_1 = r2_score(y_test, svm_predict_1)
svm_MAE_1 = mean_absolute_error(y_test, svm_predict_1)
svm_MSE_1 = mean_squared_error(y_test, svm_predict_1)
svm_RMSE_1 = mean_squared_error(y_test, svm_predict_1, squared=False)

print(f"SVM R2 Score: {svm_R2_1}")
print(f"SVM Mean Absolute Error: {svm_MAE_1}")
print(f"SVM Mean Squared Error: {svm_MSE_1}")
print(f"SVM Root Mean Squared Error: {svm_RMSE_1}")

# In[32]:


# Export the models
import joblib

joblib.dump(pipeline, 'models_new/ridge_pipeline.pkl')
joblib.dump(best_random_model, 'models_new/best_random_model.pkl')
joblib.dump(best_decision_model, 'models_new/best_decision_model.pkl')
joblib.dump(best_knn_model, 'models_new/best_knn_model.pkl')
joblib.dump(best_svm_regressor, 'models_new/best_svm_regressor.pkl')
joblib.dump(scaler, 'models_new/scaler.pkl')
joblib.dump(X_test_scaled, 'models_new/X_test_scaled.pkl')
joblib.dump(y_test, 'models_new/y_test.pkl')
joblib.dump(X_test, 'models_new/X_test.pkl')




# Print all metrics
print("R2 Score:", linear_R2)
print("R2 Score:", random_R2)
print("R2 Score:", decision_R2)
print("R2 Score:", knn_R2)
print("R2 Score:", svm_R2)



print(f"Pipeline R2 Score: {pipeline_R2}")
print(f"Random Forest R2 Score: {randomforest_R2}")
print(f"Decision Tree R2 Score: {decision_R2_1}")
print(f"KNN R2 Score: {knn_R2_1}")
print(f"SVM R2 Score: {svm_R2_1}")
