"""
Objectives¶
After completing this lab, you will be able to:

Implement and evaluate the performance of random forest regression models on real-world data
Interpret various evaluation metrics and visualizations
Describe the feature importances for a regression model
"""

"""
Introduction
In this lab, you will:

Use the California Housing data set included in scikit-learn to predict the median house price based on various attributes
Create a random forest regression model and evaluate its performance
Investigate the feature importances for the model
Your goal in this lab is not to find the best regressor - it is primarily intended for you to practice interpreting modelling results in the context of a real-world problem.
"""

!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install scipy==1.14.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

# Load the California Housing data set
data = fetch_california_housing()
X, y = data.data, data.target

# Print the description of the California Housing data set¶
print(data.DESCR)

"""
Exercise 1. Split the data into training and testing sets
Reserve 20% of the data for evaluation
"""
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42)
# The key idea is that X_train contains the input features, while y_train contains the target values you want to predict.


# Explore the training data (Create a DataFrame with column names)
df = pd.DataFrame(data = X_train, columns = data.feature_name)
# df.columns = data.feature_names
df['MedHouseVal'] = y_train
print(df.head())
print(df.describe())


# Exercise 2. What range are most of the median house prices valued at?
# answer: Considering the 25th to the 75th percentile range, most of the median house prices fall within $119,300 and $265,000.

"""
How are the median house prices distributed?
"""

plt.hist(1e5*y_train, bins = 30, color = 'lightblue', edgecolor = 'black')
plt.title(f'Median house value distribution\n Skewness: {skew(y_train):.2f}')
plt.xlabel()
plt.ylabel()
plt.show()

"""
Model fitting and prediction
Let's fit a random forest regression model to the data and use it to make median house price predicions. Use the default parameters, which includes using 100 base estimators, or regression trees.
"""
# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_regressor.predict(X_test)

""" Estimate out-of-sample MAE, MSE, RMSE, and R² """

mae = mean_absolute_error(y_train, y_pred_test)
mse = mean_squared_error(y_train, y_pred_test)
rmse = root_mean_squared_error(y_train, y_pred_test)
r2 = r2_score(y_train, y_pred_test)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

"""
Mean Absolute Error (MAE): 0.3276
Mean Squared Error (MSE): 0.2556
Root Mean Squared Error (RMSE): 0.5055
R² Score: 0.8050
"""
