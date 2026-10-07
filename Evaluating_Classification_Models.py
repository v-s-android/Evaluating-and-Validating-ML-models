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

plt.figure(figsize=(12,6))

# Original Feature Distribution (Noise-Free)
plt.subplot(1,2,1)
plt.hist(df[feature_names[5]], bins = 20, alpha = 0.7, color='blue', label = "X_scaled: original ") # or df["mean compactness"]
plt.xlabel(feature_names[5]) #  mean compactness
plt.ylabel("Frequency")

# Noisy Feature Distribution
plt.subplot(1,2,2)
plt.hist(df_noisy[feature_name[5]], bins = 20, alpha=0.7, color= 'red', label = "X_noisy")
plt.xlabel(feature_names[5]) #  mean compactness
plt.ylabel("Frequency")

plt.tight_layout() # Ensures proper spacing between subplots
plt.show()

"""
Plots
You can also plot the two features together to get a sense of their differences.
"""

plt.figure( figsize = (12,6))
plt.plot(df["mean compactness"], label="Original" , lw =3) # line width = 3
plt.plot(df_noisy["mean compactness"], label="Noisy", '--')
plt.title("Scaled feature comparison with and without noise")
plt.xlabel("mean compactness")
plt.legend()
plt.show()

"""
Scatterplot
Finally, you can compare the two features using a scatterplot. This gives you an excellent idea of how well the two features are correlated.
"""

plt.figure(figsize=(12,6))
plt.scatter(df["mean compactness"],df_noisy["mean compactness"] , lw=5 ) # line width = 5
plt.title("Scaled feature comparison with and without noise")
plt.xlabel("original")
plt.ylabel("Noisy")
plt.tight_layout()
plt.show()

### Exercise 1. Split the data, and fit the KNN and SVM models to the noisy training data

X_train, X_test, y_train, y_test = train_test_split(X_noisy, y , test_size = 0.3, random_state = 42) # from data = load_breast_cancer() X, y = data.data, data.target

knn = KNeighborsClassifier(n_neighbors = 5)
svm = SVC(kernel = 'linear', C=1, random_state = 42)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

"""
Evaluate the models
Predict on the test set
"""

y_pred_knn = knn.predict(X_test)
print("y_pred_knn", y_pred_knn)
y_pred_svm = svm.predict(X_test)
print("y_pred_svm", y_pred_svm)

"""
y_pred_knn [1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 0 0 1 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 1 0 1 0
 1 1 0 1 1 1 0 1 1 0 0 0 0 0 1 1 1 0 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 0
 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1]
y_pred_svm [1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
 1 1 1 0 1 1 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 0 1 0
 1 1 0 1 0 1 1 1 0 1 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 0
 0 1 0 1 1 1 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 0 1]
"""

# Print the accuracy scores and classification reports for both models¶

print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_svm))

"""
KNN Testing Accuracy: 0.936
SVM Testing Accuracy: 0.971

KNN Testing Data Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.89      0.91        63
           1       0.94      0.96      0.95       108

    accuracy                           0.94       171
   macro avg       0.94      0.93      0.93       171
weighted avg       0.94      0.94      0.94       171


SVM Testing Data Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96        63
           1       0.98      0.97      0.98       108

    accuracy                           0.97       171
   macro avg       0.97      0.97      0.97       171
weighted avg       0.97      0.97      0.97       171
"""

# Plot the confusion matrices

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Create a 12×5 inch figure with 2 plots arranged side-by-side, where fig represents the whole figure and axes[0] and axes[1] represent the two individual plots.
fig, axes = plt.subplot(1, 2, figsize = (12, 5))

sns.heatmap(conf_matrix_knn, annot = True ,cmap = "Blues", ax = axes[0], fmt='d', xticklabels = labels, yticklabels = labels) # labels = data.target_names from above
axes[0].set_title("KNN testing confusion matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")


sns.heatmap(conf_matrix_svm, annot = True, cmap = "Blues", ax = axes[1], fmt='d', xticklabels = labels, yticklabels = labels)
axes[1].set_title("SVM testing confusion matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

"""
Are we overfitting?¶
Let's evaluate the results on the training data and compare them against the test data results.

Exercise 4. Obtain the prediction results using the training data.
"""

y_pred_train_knn = knn.predict(X_train)
y_pred_train_svm = svm.predict(X_train)

# Evaluate the models on the training data
print(f"KNN Training Accuracy: {accuracy_score(y_train, y_pred_train_knn):.3f}")
print(f"SVM Training Accuracy: {accuracy_score(y_train, y_pred_train_svm):.3f}")

print("\nKNN Training Classification Report:")
print(classification_report(y_train, y_pred_train_knn))

print("\nSVM Training Classification Report:")
print(classification_report(y_train, y_pred_train_svm))

"""
KNN Training Accuracy: 0.955
SVM Training Accuracy: 0.972

KNN Training Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.91      0.94       149
           1       0.95      0.98      0.96       249

    accuracy                           0.95       398
   macro avg       0.96      0.95      0.95       398
weighted avg       0.96      0.95      0.95       398


SVM Training Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.95      0.96       149
           1       0.97      0.99      0.98       249

    accuracy                           0.97       398
   macro avg       0.97      0.97      0.97       398
weighted avg       0.97      0.97      0.97       398
"""

# Exercise 5. Plot the confusion matrices for the training data

# Enter your code here
conf_matrix_knn = confusion_matrix(y_train, y_pred_train_knn)
conf_matrix_svm = confusion_matrix(y_train, y_pred_train_svm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0], xticklabels=labels, yticklabels=labels)

axes[0].set_title('KNN Training Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Training Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

