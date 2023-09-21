#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> ENSEMBLE TECHNIQUES AND IT'S TYPES-5 </p>
Q1. You are working on a machine learning project where you have a dataset containing numerical and
categorical features. You have Identified that some of the features are highly correlated and there are
missing values in some of the columns. You want to build a pipeline that automates the feature
engineering process and handles the missing values.
# To build a pipeline for feature engineering that automates handling missing values and addresses highly correlated features in a machine learning project, you can use Python and libraries such as scikit-learn, pandas, and numpy. Here are the steps to create such a pipeline:
# 
# Data Preprocessing:
# Import the necessary libraries.
# Load your dataset into a pandas DataFrame.
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier  # You can choose any classifier
# 
# # Load your dataset
# data = pd.read_csv('your_dataset.csv')
# Split Data:
# Split your dataset into training and testing sets if you haven't already.
# X = data.drop('target_column', axis=1)  # Features
# y = data['target_column']  # Target variable
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Column Transformation:
# Create separate transformers for numerical and categorical features.
# Handle missing values using imputation (e.g., mean for numerical, most frequent for categorical).
# Standardize numerical features.
# Encode categorical features using one-hot encoding.
# Define transformers for numerical and categorical columns
# numerical_features = X_train.select_dtypes(include=['number']).columns
# categorical_features = X_train.select_dtypes(exclude=['number']).columns
# 
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])
# 
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
# 
# Combine transformers using ColumnTransformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numerical_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])
# Building the Pipeline:
# Create a pipeline that combines preprocessing and a classifier.
# Create the pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # You can choose any classifier
# ])
# Model Training and Evaluation:
# Fit the pipeline to your training data and evaluate its performance.
# Fit the pipeline to the training data
# pipeline.fit(X_train, y_train)
# 
# Evaluate the model on the test data
# accuracy = pipeline.score(X_test, y_test)
# print(f"Accuracy: {accuracy:.2f}")
# This pipeline automates the feature engineering process by handling missing values, standardizing numerical features, and encoding categorical features before training your machine learning model. Adjust the pipeline components and hyperparameters as needed for your specific project.

# In[ ]:




Design a pipeline that includes the following steps:
Use an automated feature selection method to Identify the important features in the dataset.
Create a numerical pipeline that includes the following steps:
Impute the missing values in the numerical columns using the mean of the column values.
Scale the numerical columns using standardisation.
Create a categor#ial pipeline that includes the following steps:
Impute the missing values in the categorical columns using the most frequent value of the column.
One-hot encode the categorical columns.
Combine the numerical and categorical pipelines using a ColumnTransformer.
Use a Random Forest Classifier to build the final model.
Evaluate the accuracy of the model on the test dataset.To design a pipeline with the specified steps, you'll need to use Python and libraries like scikit-learn for preprocessing and modeling. Here's a step-by-step implementation of the pipeline you described:
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming you have a DataFrame 'data' with both numerical and categorical features,
# and a target column 'target'.

# Step 1: Feature Selection (Use an automated method to identify important features)
# You can replace 'important_features' with your actual feature selection method.
important_features = ['feature1', 'feature2', 'feature3']

# Separate features and target variable
X = data[important_features]
y = data['target']

# Step 2: Create numerical and categorical pipelines
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Step 3: Combine numerical and categorical pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Step 4: Create the final pipeline with a Random Forest Classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Fit the model on the training data
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
Make sure to replace 'feature1', 'feature2', 'feature3', and 'target' with the actual names of your features and target variable. This pipeline will preprocess the data, including feature selection, imputation, scaling, and one-hot encoding for numerical and categorical features, and then fit a Random Forest Classifier to the preprocessed data. Finally, it evaluates the model's accuracy on the test dataset.
# In[ ]:





# Q2. Build a pipeline that includes a random forest classifier and a logistic regression classifier, and then
# use a voting classifier to combine their predictions. Train the pipeline on the iris dataset and evaluate its
# accuracy.
To build a pipeline that includes both a Random Forest Classifier and a Logistic Regression Classifier, and then use a Voting Classifier to combine their predictions on the Iris dataset, you can follow these steps:
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual classifiers
rf_classifier = RandomForestClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42)

# Create a Voting Classifier that combines the predictions
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', rf_classifier),
        ('lr', lr_classifier)
    ],
    voting='hard'  # Use 'hard' voting for classification
)

# Create a pipeline with scaling (optional)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # You can omit this step if you want
    ('voting_classifier', voting_classifier)
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the accuracy of the ensemble model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
In this code:

We load the Iris dataset and split it into training and testing sets.
We create individual classifiers, one for Random Forest and one for Logistic Regression.
We create a Voting Classifier (voting_classifier) that combines the predictions of these two classifiers using hard voting.
Optionally, we add a StandardScaler to scale the data within the pipeline.
We train the pipeline on the training data.
We make predictions on the test data and calculate the accuracy of the ensemble model.
You can adjust the classifiers and voting type as needed for your specific use case.
# In[ ]:





# #  <P style="color:green"> THAT'S ALL, THANK YOU    </p>
