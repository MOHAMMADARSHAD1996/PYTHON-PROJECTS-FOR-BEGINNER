#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> ENSEMBLE TECHNIQUES AND IT'S TYPES-4  </p>
Build a random forest classifier to predict the risk of heart disease based on a dataset of patient 
information. The dataset contains 303 instances with 14 features, including age, sex, chest pain type, 
resting blood pressure, serum cholesterol, and maximum heart rate achieved.

Dataset link: https://drive.google.com/file/d/1bGoIE4Z2kG5nyh-fGZAJ7LH0ki3UfmSJ/view?
usp=share_linkQ1. Preprocess the dataset by handling missing values, encoding categorical variables, and scaling the
numerical features if necessary.To preprocess the dataset you provided, which appears to be a medical dataset with features like age, sex, cholesterol levels (chol), and a target variable, you can follow these steps:

Handling Missing Values:

Check for missing values in the dataset for each column.
Decide on an appropriate strategy to handle missing values based on the nature of the data. Since the dataset is not specified, I'll assume some common strategies:
If there are missing values in numerical features like age, trestbps, chol, thalach, oldpeak, you can impute them with the mean or median of the respective column.
For categorical features like sex, cp, fbs, restecg, exang, slope, ca, and thal, you may consider encoding missing values as a separate category or impute them using the mode (most frequent category).
Encoding Categorical Variables:

Encode categorical variables into numerical format so that machine learning algorithms can work with them. Common techniques include:
One-Hot Encoding: Create binary columns for each category within a categorical feature.
Label Encoding: Assign a unique integer to each category.
Ordinal Encoding: Assign integers based on the ordinal relationship between categories (if applicable).
Scaling Numerical Features (if necessary):

Scaling numerical features is essential for some machine learning algorithms (e.g., K-Nearest Neighbors, Support Vector Machines) to ensure that all features have the same scale. Common scaling methods include:
Min-Max Scaling (Normalization): Scales features to a specific range, typically between 0 and 1.
Standardization (Z-score Scaling): Scales features to have a mean of 0 and a standard deviation of 1.
Robust Scaling: Scales features using statistics that are robust to outliers.
Here's a Python code example using the popular library pandas for handling missing values and encoding categorical variables:
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('your_dataset.csv')

# Handle missing values for numerical features
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Handle missing values for categorical features (assuming they are object data types)
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical variables using Label Encoding (you can use One-Hot Encoding as well)
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Scale numerical features using Standardization
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Now, df contains the preprocessed dataset.
Remember to customize the code based on the specifics of your dataset and the preprocessing strategies you want to apply.
# In[ ]:





# Q2. Split the dataset into a training set (70%) and a test set (30%).
To split your dataset into a training set (70%) and a test set (30%), you can use Python's train_test_split function from the sklearn.model_selection module. Here's how you can do it:
from sklearn.model_selection import train_test_split

# Assuming you have already loaded and preprocessed your dataset as 'df'

# Define the features (X) and the target variable (y)
X = df.drop('target', axis=1)  # Features
y = df['target']              # Target variable

# Split the dataset into a training set and a test set (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 'X_train' and 'y_train' are the training features and labels.
# 'X_test' and 'y_test' are the test features and labels.
In this code:

X contains all the features except the 'target' column.
y contains the 'target' column, which is your target variable.
train_test_split is used to split the data into a training set (70%) and a test set (30%).
The random_state parameter is set to ensure reproducibility. You can change this value or omit it if you want different random splits each time you run the code.
Now, you have your dataset split into training and test sets, which you can use for training and evaluating machine learning models.
# In[ ]:





# Q3. Train a random forest classifier on the training set using 100 trees and a maximum depth of 10 for each
# tree. Use the default values for other hyperparameters.
To train a Random Forest Classifier on the training set with 100 trees and a maximum depth of 10 for each tree using Python's scikit-learn library, you can follow these steps:
from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest Classifier with specified hyperparameters
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier on the training set
rf_classifier.fit(X_train, y_train)

# Once trained, you can use this classifier for predictions on new data.
Here's what each step does:

Import the RandomForestClassifier class from the sklearn.ensemble module.

Create an instance of the RandomForestClassifier class with the specified hyperparameters:

n_estimators: The number of trees in the forest (100 in this case).
max_depth: The maximum depth of each tree (10 in this case).
random_state: A seed for the random number generator for reproducibility.
Train the classifier on the training set using the .fit() method. X_train contains the training features, and y_train contains the corresponding labels.

After training, your rf_classifier is ready to make predictions on new data. You can use it to evaluate its performance on the test set or for other classification tasks.
# In[ ]:





# Q4. Evaluate the performance of the model on the test set using accuracy, precision, recall, and F1 score.
To evaluate the performance of your Random Forest Classifier on the test set using accuracy, precision, recall, and F1 score, you can use scikit-learn's metrics module. Here's how you can calculate these metrics:

python
Copy code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
Here's what each step does:

Use the trained rf_classifier to make predictions on the test set features X_test.

Calculate accuracy using accuracy_score. It measures the proportion of correctly predicted instances.

Calculate precision using precision_score. It measures the ratio of true positive predictions to the total positive predictions.

Calculate recall using recall_score. It measures the ratio of true positive predictions to the total actual positives.

Calculate the F1 score using f1_score. It's the harmonic mean of precision and recall and provides a balance between the two metrics.

Finally, print the values of accuracy, precision, recall, and F1 score to evaluate the model's performance on the test set.
# In[ ]:





# Q5. Use the feature importance scores to identify the top 5 most important features in predicting heart
# disease risk. Visualise the feature importances using a bar chart.
You can use the feature importance scores provided by the trained Random Forest Classifier to identify the top 5 most important features in predicting heart disease risk and visualize them using a bar chart. Here's how you can do that:

import matplotlib.pyplot as plt
import numpy as np

# Get feature importances from the trained classifier
feature_importances = rf_classifier.feature_importances_

# Get the names of the features
feature_names = X.columns

# Sort feature importances in descending order and get the indices
sorted_indices = np.argsort(feature_importances)[::-1]

# Get the top 5 most important features
top_5_indices = sorted_indices[:5]
top_5_features = [feature_names[i] for i in top_5_indices]
top_5_importances = [feature_importances[i] for i in top_5_indices]

# Create a bar chart to visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_5_features, top_5_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Top 5 Most Important Features for Heart Disease Prediction')
plt.gca().invert_yaxis()  # Invert the y-axis to display the most important feature at the top
plt.show()
In this code:

We retrieve the feature importances using the feature_importances_ attribute of the trained rf_classifier.

We get the names of the features from the original dataset.

We sort the feature importances in descending order and get the indices of the sorted features.

We select the top 5 most important features by taking the first 5 indices from the sorted list.

We create a horizontal bar chart using matplotlib to visualize the feature importances. The most important feature will be at the top of the chart.

Running this code will display a bar chart showing the relative importance of the top 5 features in predicting heart disease risk.
# In[ ]:





# Q6. Tune the hyperparameters of the random forest classifier using grid search or random search. Try
# different values of the number of trees, maximum depth, minimum samples split, and minimum samples
# leaf. Use 5-fold cross-validation to evaluate the performance of each set of hyperparameters.
Tuning the hyperparameters of the Random Forest Classifier can significantly improve its performance. You can use grid search or random search in combination with cross-validation to find the best hyperparameters. Here's how you can perform hyperparameter tuning using grid search and 5-fold cross-validation:
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Get the best model with the best hyperparameters
best_rf_classifier = grid_search.best_estimator_

# Evaluate the best model on the test set
best_model_accuracy = best_rf_classifier.score(X_test, y_test)

print("Best Hyperparameters:", best_params)
print("Best Model Accuracy:", best_model_accuracy)
In this code:

We create a RandomForestClassifier and specify a range of hyperparameter values to search through using the param_grid dictionary.

We use GridSearchCV to perform a grid search with 5-fold cross-validation. It tries all possible combinations of hyperparameters specified in param_grid.

The best hyperparameters are obtained using best_params_, and the best model is obtained using best_estimator_.

We evaluate the best model's accuracy on the test set.

You can modify the param_grid dictionary to include other hyperparameters you want to tune or expand the range of values to search for. Additionally, you can use a different scoring metric (besides accuracy) if it better suits your problem.
# In[ ]:





# Q7. Report the best set of hyperparameters found by the search and the corresponding performance
# metrics. Compare the performance of the tuned model with the default model.
To report the best set of hyperparameters found by the grid search and the corresponding performance metrics, and to compare the performance of the tuned model with the default model, you can use the following code:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the default Random Forest Classifier
default_rf_classifier = RandomForestClassifier(random_state=42)

# Train the default model on the training set
default_rf_classifier.fit(X_train, y_train)

# Evaluate the default model on the test set
y_pred_default = default_rf_classifier.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)

# Report the performance of the default model
print("Default Model Accuracy:", accuracy_default)

# Print the best hyperparameters found by the grid search
print("Best Hyperparameters:", best_params)

# Evaluate the best model on the test set
y_pred_best = best_rf_classifier.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

# Report the performance of the tuned model
print("Tuned Model Accuracy:", accuracy_best)
print("Tuned Model Precision:", precision_best)
print("Tuned Model Recall:", recall_best)
print("Tuned Model F1 Score:", f1_best)
In this code:

We first define and train the default Random Forest Classifier using the same training data as before.

We evaluate the default model's accuracy on the test set.

We report the performance of the default model.

We print the best hyperparameters found by the grid search, which were stored in the best_params variable.

We evaluate the best model (tuned model) on the test set and calculate accuracy, precision, recall, and F1 score.

We report the performance of the tuned model.

By comparing the performance metrics of the default model with those of the tuned model, you can assess whether the hyperparameter tuning improved the model's predictive performance. Typically, you would expect the tuned model to perform better than the default model, but the degree of improvement can vary depending on the dataset and the choice of hyperparameters.
# In[ ]:





# Q8. Interpret the model by analysing the decision boundaries of the random forest classifier. Plot the
# decision boundaries on a scatter plot of two of the most important features. Discuss the insights and
# limitations of the model for predicting heart disease risk.
Interpreting the decision boundaries of a Random Forest Classifier can provide valuable insights into how the model makes predictions. However, since the dataset you provided has multiple features, it's challenging to visualize all the decision boundaries in a single plot. Therefore, we'll focus on visualizing the decision boundaries on a scatter plot of two of the most important features.

Let's assume we choose two features, "age" and "thalach" (maximum heart rate achieved), for visualization. Here's how you can do it:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Select two features for visualization (e.g., 'age' and 'thalach')
feature1 = 'age'
feature2 = 'thalach'

# Extract the corresponding columns from the dataset
X_visualize = X_train[[feature1, feature2]].values

# Get the corresponding feature indices
feature1_idx = X.columns.get_loc(feature1)
feature2_idx = X.columns.get_loc(feature2)

# Train the Random Forest Classifier on these two features
rf_classifier.fit(X_visualize, y_train)

# Create a mesh grid to plot decision boundaries
x_min, x_max = X_visualize[:, 0].min() - 1, X_visualize[:, 0].max() + 1
y_min, y_max = X_visualize[:, 1].min() - 1, X_visualize[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict class labels for each point in the mesh grid
Z = rf_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a scatter plot of the data points
plt.figure(figsize=(10, 6))
cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.4)

# Plot the data points
scatter = plt.scatter(X_visualize[:, 0], X_visualize[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=100)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title("Decision Boundaries of Random Forest Classifier")

# Create a legend
handles, labels = scatter.legend_elements()
plt.legend(handles, ["No Heart Disease", "Heart Disease"])

plt.show()
In this code:

We select two features, "age" and "thalach," and extract them from the training data.

We train the Random Forest Classifier on these two features.

We create a mesh grid of points covering the range of the two features to visualize the decision boundaries.

We predict class labels for each point in the mesh grid and plot the decision boundaries using contourf.

We create a scatter plot of the data points, where points are colored based on their true labels.

Interpreting the plot:

The decision boundaries separate the feature space into regions corresponding to different predicted classes (heart disease or no heart disease).
Areas with the same color represent regions where the classifier predicts the same class.
The model's decision boundaries are non-linear, which suggests that it can capture complex relationships between the features and the target variable.
Limitations and Insights:

The visualization provides insights into how the Random Forest Classifier separates data points based on "age" and "thalach." However, it's limited to a two-dimensional space and cannot capture interactions between all features.
The model appears to perform well in distinguishing between some data points with heart disease and those without it in this two-feature space. However, evaluating its overall performance requires considering all features and using appropriate metrics.
To gain a more comprehensive understanding of the model's strengths and limitations, you should also analyze feature importances, conduct feature engineering, and evaluate the model using various performance metrics and validation techniques on the entire dataset.
# In[ ]:





# #  <P style="color:green"> THAT'S ALL, THANK YOU    </p>
