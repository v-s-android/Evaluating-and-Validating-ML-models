"""
Objectives

Implement and evaluate the performance of classification models on real-world data
Interpret and compare various evaluation metrics and the confusion matrix for each model
"""

"""
Introduction
In this lab, you will:

Use the breast cancer data set included in scikit-learn to predict whether a tumor is benign or malignant
Create two classification models and evaluate them.
Add some Gaussian random noise to the features to simulate measurement errors
Interpreting and comparing the various evaluation metrics and the confusion matrix for each model will provide you with some valuable intuition regarding what the evaluation metrics mean and how they might impact your interpretation of the model performances.

Your goal in this lab is not to find the best classifier - it is primarily intended for you to practice interpreting and comparing results in the context of a real-world problem.
"""
!pip install numpy==2.2.0
!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Breast Cancer data set
data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
print("labels: ",labels)
feature_names = data.feature_names
print("feature_names ", feature_names)
"""
labels:  ['malignant' 'benign']
feature_names:  ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
 """

# Print the description of the Breast Cancer data set
print(data.DESCR)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# Add some noise
# Next, add some noise to simulate random measurement error, then view the first few rows of the original and noisy features for comparison.
# Add Gaussian noise to the data set
np.random.seed(42)
noise_factor = 0.5
X_noisy = X_scaled + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = X.shape )

# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled , columns = feature_names)
df_noisy = pd.DataFrame(X_noisy , columns = feature_names)

print(df.head())

print(df_noisy.head())

"""
Visualizing the noise content.
You can get a good idea of how much noise there is in the features by comparing values in the previous tables.
You can also visualize the differences in several ways. Let's begin by plotting the histograms of one of the features with and without noise for comparison.
"""
