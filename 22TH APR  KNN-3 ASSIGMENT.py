#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> KNN-3 </p>

# Q1. Write a Python code to implement the KNN classifier algorithm on load_iris dataset in
# sklearn.datasets.
To implement the K-Nearest Neighbors (KNN) classifier algorithm on the load_iris dataset from sklearn.datasets, you can follow these steps. First, make sure you have scikit-learn installed (pip install scikit-learn).

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN classifier with a chosen 'k' value (e.g., k=3)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
We import the necessary libraries, including the iris dataset, train-test split, KNeighborsClassifier, and accuracy_score.
We load the iris dataset, which contains features (X) and target labels (y).
We split the dataset into training and testing sets using train_test_split.
We initialize the KNN classifier with a chosen 'k' value (in this example, k=3).
We fit the KNN classifier to the training data using fit.
We make predictions on the test data using predict.
We calculate the accuracy of the model using accuracy_score by comparing the predicted labels (y_pred) to the actual labels (y_test) and print the accuracy.
You can adjust the value of 'k' and other hyperparameters to fine-tune the model's performance based on your specific requirements.
# In[ ]:





# Q2. Write a Python code to implement the KNN regressor algorithm on load_boston dataset in
# sklearn.datasets.
To implement the K-Nearest Neighbors (KNN) regressor algorithm on the load_boston dataset from sklearn.datasets, you can follow these steps. Make sure you have scikit-learn installed (pip install scikit-learn).
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the boston dataset
boston = load_boston()
X = boston.data  # Features
y = boston.target  # Target variable (housing prices)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN regressor with a chosen 'k' value (e.g., k=3)
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# Fit the regressor to the training data
knn_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_regressor.predict(X_test)

# Calculate the mean squared error (MSE) and R-squared (R2) score of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")
We import the necessary libraries, including the load_boston dataset, train-test split, KNeighborsRegressor, and evaluation metrics.
We load the load_boston dataset, which contains features (X) and target values (y) representing housing prices.
We split the dataset into training and testing sets using train_test_split.
We initialize the KNN regressor with a chosen 'k' value (in this example, k=3).
We fit the KNN regressor to the training data using fit.
We make predictions on the test data using predict.
We calculate the mean squared error (MSE) and R-squared (R2) score of the model using mean_squared_error and r2_score, respectively, and print these evaluation metrics.
You can adjust the value of 'k' and other hyperparameters to fine-tune the model's performance based on your specific regression task.
# In[ ]:





# Q3. Write a Python code snippet to find the optimal value of K for the KNN classifier algorithm using
# cross-validation on load_iris dataset in sklearn.datasets.

To find the optimal value of K for the K-Nearest Neighbors (KNN) classifier algorithm using cross-validation on the load_iris dataset from sklearn.datasets, you can perform a grid search over different values of K and evaluate the model's performance using cross-validation. Here's a Python code snippet to do that:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a KNN classifier
knn_classifier = KNeighborsClassifier()

# Define a range of K values to search
param_grid = {'n_neighbors': range(1, 21)}  # Adjust the range as needed

# Create a grid search object with cross-validation
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

# Perform the grid search to find the optimal K value
grid_search.fit(X_train, y_train)

# Get the best K value from the grid search
best_k = grid_search.best_params_['n_neighbors']

# Print the best K value and corresponding cross-validation score
print(f"Optimal K: {best_k}")
print(f"Cross-validation Accuracy: {grid_search.best_score_:.4f}")

# Fit the KNN classifier with the best K value to the training data
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
best_knn_classifier.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = best_knn_classifier.score(X_test, y_test)
print(f"Test Accuracy with Optimal K: {accuracy:.4f}
We load the iris dataset and split it into training and testing sets.
We create a KNN classifier and define a range of K values (from 1 to 20 in this example) to search using grid search.
We set up a grid search object (GridSearchCV) with 5-fold cross-validation and specify 'accuracy' as the scoring metric.
We perform the grid search to find the optimal K value by fitting the model to the training data.
We extract the best K value and its corresponding cross-validation accuracy from the grid search results.
We create a new KNN classifier with the best K value and fit it to the training data.
Finally, we evaluate the model's accuracy on the test data using the optimal K value.
You can adjust the range of K values and other hyperparameters in the param_grid dictionary to fine-tune the grid search.
# In[ ]:





# Q4. Implement the KNN regressor algorithm with feature scaling on load_boston dataset in
# sklearn.datasets.
To implement the K-Nearest Neighbors (KNN) regressor algorithm with feature scaling on the load_boston dataset from sklearn.datasets, you can follow these steps. First, make sure you have scikit-learn installed (pip install scikit-learn).

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the boston dataset
boston = load_boston()
X = boston.data  # Features
y = boston.target  # Target variable (housing prices)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the StandardScaler for feature scaling
scaler = StandardScaler()

# Fit and transform the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN regressor with a chosen 'k' value (e.g., k=3)
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# Fit the regressor to the scaled training data
knn_regressor.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = knn_regressor.predict(X_test_scaled)

# Calculate the mean squared error (MSE) and R-squared (R2) score of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")
In this code:

We import the necessary libraries, including the load_boston dataset, train-test split, KNeighborsRegressor, StandardScaler for feature scaling, and evaluation metrics.
We load the load_boston dataset, which contains features (X) and target values (y) representing housing prices.
We split the dataset into training and testing sets using train_test_split.
We initialize a StandardScaler to scale the features.
We fit and transform the scaler on the training data, ensuring that the mean of each feature is centered at 0 and the variance is 1.
We transform the test data using the same scaler to ensure consistent scaling.
We initialize the KNN regressor with a chosen 'k' value (in this example, k=3).
We fit the KNN regressor to the scaled training data using fit.
We make predictions on the scaled test data using predict.
We calculate the mean squared error (MSE) and R-squared (R2) score of the model using mean_squared_error and r2_score, respectively, and print these evaluation metrics.
By scaling the features, we ensure that each feature contributes equally to distance calculations and help the KNN regressor perform effectively on the load_boston dataset.
# In[ ]:





# Q5. Write a Python code snippet to implement the KNN classifier algorithm with weighted voting on
# load_iris dataset in sklearn.datasets.
To implement the K-Nearest Neighbors (KNN) classifier algorithm with weighted voting on the load_iris dataset from sklearn.datasets, you can follow these steps. Weights can be assigned to neighbors based on their distances to the data point of interest. Here's a Python code snippet to do that:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN classifier with a chosen 'k' value (e.g., k=3) and weights set to 'distance'
knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')

# Fit the classifier to the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
We import the necessary libraries, including the load_iris dataset, train-test split, KNeighborsClassifier, and accuracy_score for evaluation.
We load the load_iris dataset, which contains features (X) and target labels (y).
We split the dataset into training and testing sets using train_test_split.
We initialize the KNN classifier with a chosen 'k' value (in this example, k=3) and set the weights parameter to 'distance'. Setting weights to 'distance' means that closer neighbors will have more influence on the prediction than farther neighbors.
We fit the KNN classifier to the training data using fit.
We make predictions on the test data using predict.
We calculate the accuracy of the model using accuracy_score by comparing the predicted labels (y_pred) to the actual labels (y_test) and print the accuracy.
By setting the weights parameter to 'distance', we implement weighted voting in the KNN classifier, where closer neighbors have a stronger influence on the final prediction.
# In[ ]:





# Q6. Implement a function to standardise the features before applying KNN classifier.
To standardize the features before applying a K-Nearest Neighbors (KNN) classifier, you can create a function that uses StandardScaler from scikit-learn to scale the features. Standardization ensures that all features have a mean of 0 and a standard deviation of 1, making them comparable in terms of scale. Here's a Python function to standardize the features:

from sklearn.preprocessing import StandardScaler

def standardize_features(X_train, X_test):
    """
    Standardizes the features of both the training and test datasets.

    Parameters:
    X_train (array-like): Training features.
    X_test (array-like): Test features.

    Returns:
    X_train_scaled (array-like): Standardized training features.
    X_test_scaled (array-like): Standardized test features.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the scaler on the training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform the test data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
You can use this function to standardize the features before training your KNN classifier. Here's how you can use it:
# Load the iris dataset and split it into training and testing sets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

# Initialize and fit the KNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = knn_classifier.predict(X_test_scaled)
This way, you standardize the features in both the training and test datasets, ensuring that your KNN classifier works with standardized data.
# In[ ]:





# Q7. Write a Python function to calculate the euclidean distance between two points.
You can create a Python function to calculate the Euclidean distance between two points in n-dimensional space. Here's a simple function to do that:

import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    point1 (array-like): The coordinates of the first point.
    point2 (array-like): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    # Convert the points to NumPy arrays to handle different dimensions easily
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate the Euclidean distance using NumPy's vectorized operations
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    
    return distance
You can use this function to calculate the Euclidean distance between two points by providing their coordinates as input. Here's an example of how to use the function:

# Example usage
point1 = [1, 2, 3]
point2 = [4, 5, 6]

distance = euclidean_distance(point1, point2)
print(f"Euclidean Distance: {distance}")
This function is flexible and can handle points in different dimensions. It calculates the Euclidean distance as the square root of the sum of squared differences between corresponding coordinates of the two points.
# In[ ]:





# Q8. Write a Python function to calculate the manhattan distance between two points.
You can create a Python function to calculate the Manhattan distance (also known as the L1 distance) between two points in n-dimensional space. Here's a function to calculate the Manhattan distance:
import numpy as np

def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.

    Parameters:
    point1 (array-like): The coordinates of the first point.
    point2 (array-like): The coordinates of the second point.

    Returns:
    float: The Manhattan distance between the two points.
    """
    # Convert the points to NumPy arrays to handle different dimensions easily
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Calculate the Manhattan distance using NumPy's vectorized operations
    distance = np.sum(np.abs(point1 - point2))
    
    return distance
You can use this function to calculate the Manhattan distance between two points by providing their coordinates as input. Here's an example of how to use the function:

# Example usage
point1 = [1, 2, 3]
point2 = [4, 5, 6]

distance = manhattan_distance(point1, point2)
print(f"Manhattan Distance: {distance}")
This function calculates the Manhattan distance as the sum of the absolute differences between corresponding coordinates of the two points.
# In[ ]:





# #  <P style="color:green"> THAT'S ALL, THANK YOU    </p>
