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

"""
Exercise 3. What do these statistics mean to you?
How comfortable could you be with stopping here and communicating the results to the C-suite?

Answer: 
The mean absolute error is $33,220.

So, on average, predicted median house prices are off by $33k.

Mean squared error is less intuitive to interpret, but is usually what is being minimized by the model fit.
On the other hand, taking the square root of MSE yields a dollar value, here RMSE = $50,630.

An R-squared score of 0.80 is not considered very high. It means the model explains about %80 of the variance in median house prices,
although this interpretation can be misleading for compex data with nonlinear relationships, skewed values, and outliers. R-squard can still be useful for comparing models though.

These statistics alone don't explain any details about the performance of the model. For example, where did the model do well or poorly?
"""

# Plot Actual vs Predicted values

plt.scatter(y_test, y_pred_test, color = 'blue', alpha = 0.5) # This code is used to compare the actual target values (y_test) with the model's predicted values (y_pred_test).
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # A true perfect-prediction reference line, K= black, -- = dash line
plt.xlabel("Acutal values")
plt.ylabel("Predicted values")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.show()

"""
Exercise 4. Plot the histogram of the residual errors (dollars)
Also, print the mean and standard deviation of the residuals. Check for any patterns.
"""
residuals = 1e5 * (y_test - y_pred_test) #calculates the residuals (prediction errors) of your model 

"""
y_test: Actual values from your test dataset.

y_pred_test: Values predicted by your model.

y_test - y_pred_test: Calculates the prediction error for each observation.

1e5: Means 1×10^5 =100,000. It multiplies each residual by 100,000 to make the values easier to view on the chosen scale.
"""

# plot the histogram of the residuals
plt.hist(residuals, bins = 30, color = 'lightblue', edgecolor = 'black')
plt.title("Median House Value Prediction Residuals")
plt.xlabel("Median House Value Prediction Error ($)")
plt.ylabel("Frequency")
plt.show()

print("Average error(mean of residuals): ",str(int(np.mean(residuals))))
print("Standard deviation of error: ", str(int(np.std(residuals))))
"""
Average error = -1215
Standard deviation of error = 50537
"""

"""
Exercise 5. Plot the model residual errors by median house value.
Sort the residuals by actual median house value before plotting the residuals.

Check for any patterns.
"""

# Create a DataFrame to make sorting easy
residuals_df = pd.DataFrame({ 'Actual' : 1e5 * y_test, 'Residuals': residuals })
print(residuals_df.head())
# Sort the DataFrame by the actual target values
residuals_df = residuals_df.sort_values( by = 'Actual')

# Plot the residuals
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], alpha = 0.5, ec='k' ) # k is black, marker = 'o',
plt.title('Median House Value Prediciton Residuals Ordered by Actual Median Prices')
plt.xlabel('Actual Values (Sorted)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

""" residuals_df
     Actual  Residuals
0   47700.0   -3245.00
1   45800.0  -28705.00
2  500001.0    7675.29
3  218600.0  -34246.00
4  278000.0   50097.00
"""

### Exercise 6. What trend can you infer from this residual plot?
""" Although we saw a small average residual of only -$1400, you can see from this plot that the average error as a function of median house price is actually increasing
from negative to positive values. In other words, lower median prices tend to be overpredicted while higher median prices tend to be underpredicted."""

"""
Exercise 7. Display the feature importances as a bar chart.
Do you think these feature weights have practial significance? Are any of the features possibly sharing importance with other correlated features?
"""

# Feature importances
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

# Plot feature importances
plt.bar(range(X.shape[1]), importances[indices],  align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.show()

