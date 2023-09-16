#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple">   ML PROJECT </p>

# Create and deploy a ML project by i\porting load_breast_cancer dataset fro\ sklearnIload_dataset and
# apply the following:
Create a folder in which you want to create the project, after that use the git init and the necessary commands to create the specific Git repository.To create and deploy an ML project using the Breast Cancer dataset from scikit-learn, you can follow these steps:

Step 1: Set up Your Project Folder

First, create a folder for your project. You can do this using the command line:

mkdir breast_cancer_ml_project
cd breast_cancer_ml_project
Step 2: Initialize a Git Repository

Next, initialize a Git repository in your project folder:

git init
Step 3: Create Your Python Script

Now, create a Python script for your machine learning project. You can use your preferred text editor or IDE. Create a file named breast_cancer_ml.py and add your code. Here's a sample code to load the Breast Cancer dataset and perform some basic operations:

# breast_cancer_ml.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Perform your machine learning operations here
# ...

# Save and export your model or results as needed
# ...
Step 4: Commit Your Code to Git

Add your Python script to the Git repository and commit your changes:

git add breast_cancer_ml.py
git commit -m "Initial commit"
Step 5: Create a .gitignore File

Create a .gitignore file to specify which files and directories should be excluded from version control. For example, you might want to exclude Python virtual environments and data files. Create a .gitignore file with the following content:

*.pyc
__pycache__
venv/
data/
Step 6: Push Your Repository to GitHub (Optional)

If you want to host your Git repository on GitHub, create a new repository on GitHub and then push your local repository to it. Replace your-username and your-repo-name with your GitHub username and repository name:

git remote add origin https://github.com/your-username/your-repo-name.git
git branch -M main
git push -u origin main
Step 7: Install Required Dependencies

Make sure to install the required Python dependencies for your project. You can create a requirements.txt file to list them. For example:

scikit-learn==0.24.2
numpy==1.19.5
pandas==1.1.5
You can install these dependencies using pip:

pip install -r requirements.txt
Step 8: Create Documentation (Optional)

Consider adding documentation to your project, such as a README file explaining the purpose of your project, how to run it, and any other relevant information.

Now, you have a basic setup for your machine learning project using the Breast Cancer dataset from scikit-learn, and it's version-controlled with Git. You can continue to develop your project, experiment with machine learning models, and push your changes to the Git repository as needed.
# In[ ]:





# Create a separate environment so that you do not mess up with your base environment.
Creating a separate virtual environment for your machine learning project is a good practice to keep your project dependencies isolated and prevent conflicts with your base environment. You can use virtualenv or conda to create a virtual environment. Here, I'll provide instructions for both methods:

Using virtualenv:

Install virtualenv if you haven't already:
pip install virtualenv
Create a virtual environment for your project:
virtualenv breast_cancer_env
This command will create a new virtual environment named "breast_cancer_env" in your project folder.

Activate the virtual environment:

On Windows:
.\breast_cancer_env\Scripts\activate
On macOS and Linux:
source breast_cancer_env/bin/activate
You should see the name of your virtual environment in your terminal prompt, indicating that you are now working within the virtual environment.

Install the required dependencies into this environment. You can use the pip install -r requirements.txt command to install the dependencies mentioned in your requirements.txt file.

Now, you can run your Python script and work on your machine learning project within this isolated environment.

To deactivate the virtual environment and return to your base environment, simply use the deactivate command.

Using conda:

If you have Anaconda or Miniconda installed, you can create a new conda environment for your project. If not, you can install Miniconda from https://docs.conda.io/en/latest/miniconda.html.

Create a new conda environment:
conda create --name breast_cancer_env python=3.8
This command will create a new conda environment named "breast_cancer_env" with Python 3.8 installed.

Activate the conda environment:
conda activate breast_cancer_env
Now, you are working within the "breast_cancer_env" conda environment.

Install the required dependencies into this environment using pip or conda:
pip install -r requirements.txt
Run your Python script and work on your machine learning project within this isolated conda environment.
To deactivate the conda environment and return to your base environment, use the conda deactivate command.

By following these steps, you'll have a separate virtual environment for your machine learning project, ensuring that your project dependencies don't interfere with your base environment
# In[ ]:





# Create the folder structure/directories and files using the python programme required for a ML project.
# You can refer the following project structure:
# - src
# - __init__.py
# - logger.py
# - exception.py
# - utils.py
# - cmponents
# - __init__.py
# - data_ingestion.py
# - data_transformation.py
# - model_trainer.py
# - pipelin4
# - __init__.py
# - predict_pipeline.py
# - train_pipeline.py
# - import_data.py
# - setup.py
# - notebook
# - require'ents.txt
Here's a Python program to create the folder structure, directories, and files for your ML project based on the provided structure:

import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define the project structure
project_structure = {
    'src': ['__init__.py', 'logger.py', 'exception.py', 'utils.py'],
    'components': ['__init__.py', 'data_ingestion.py', 'data_transformation.py', 'model_trainer.py'],
    'pipelines': ['__init__.py', 'predict_pipeline.py', 'train_pipeline.py'],
    'import_data.py',
    'setup.py',
    'notebook',
    'requirements.txt'
}

# Create the project directory
project_name = 'your_ml_project'  # Replace with your project name
if not os.path.exists(project_name):
    os.makedirs(project_name)

# Create the project structure
for directory, files in project_structure.items():
    dir_path = os.path.join(project_name, directory)
    create_directory(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'w') as f:
            pass

print(f'Project structure created for "{project_name}"')
Replace 'your_ml_project' with the name of your project folder. When you run this program, it will create the specified directory structure and the necessary Python files with placeholder content in your project folder.

After running this code, your project directory will have the structure and files you provided:

your_ml_project/
|-- src/
|   |-- __init__.py
|   |-- logger.py
|   |-- exception.py
|   |-- utils.py
|-- components/
|   |-- __init__.py
|   |-- data_ingestion.py
|   |-- data_transformation.py
|   |-- model_trainer.py
|-- pipelines/
|   |-- __init__.py
|   |-- predict_pipeline.py
|   |-- train_pipeline.py
|-- import_data.py
|-- setup.py
|-- notebook/
|-- requirements.txt
You can then start adding your code and content to the respective files and directories as needed for your ML project.
# In[ ]:





# After this update the created folders and files to your git repository by pushing from your end and add
# following files from github and pull it to your source code:
# , RADME.md
# , LICENCE
# , .gitignorE
I can guide you on how to push the created folders and files to your Git repository and add the specified files from GitHub. Please follow these steps:

Step 1: Push the Created Folders and Files to Your Git Repository

Before pushing your files to a Git repository, you need to initialize a Git repository in your project directory (if you haven't already). Assuming you've already initialized the repository as per the previous instructions, here's how you can push the code:

# Change directory to your project folder
cd your_ml_project

# Add all files to the Git staging area
git add .

# Commit your changes
git commit -m "Initial project structure"

# Add the remote repository URL (replace with your repository URL)
git remote add origin <repository_url>

# Push the code to the remote repository (replace 'main' with your branch name)
git push -u origin main
Replace <repository_url> with the URL of your Git repository.

Step 2: Add the Specified Files from GitHub

To add the specified files (README.md, LICENSE, .gitignore) from a GitHub repository to your source code, follow these steps:

Go to the GitHub repository where these files are located.

Click on the file you want to add (e.g., README.md).

Click the "Raw" button to view the raw content of the file.

Copy the URL from your browser's address bar.

In your local project directory (where you created your ML project), use curl or wget to download the files. For example, to download README.md:
curl -o README.md <raw_file_url>
Replace <raw_file_url> with the URL you copied in step 4.

Repeat steps 3-5 for the other files (LICENSE and .gitignore) if needed.

After downloading the files, add them to your Git repository:
git add README.md LICENSE .gitignore
git commit -m "Add README, LICENSE, and .gitignore"
Push the changes to your remote repository (if it's not already done):
git push
Now, you've added the specified files from GitHub to your local project, and they are part of your Git repository. You can modify and commit these files as needed for your project
# In[ ]:





# write the program for setup.py and the relevant dependencies in requirements.txt and generate
# egg.info folder.
Certainly! Here's a sample setup.py file for your Python project along with a requirements.txt file that lists relevant dependencies. Additionally, I'll explain how to generate an egg-info folder.
setup.py:
from setuptools import setup, find_packages

setup(
    name="your_ml_project",
    version="0.1",
    author="Your Name",
    author_email="your@email.com",
    description="Description of your ML project",
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
        "numpy",
        "scikit-learn",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
In the setup.py file:

Replace "your_ml_project" with the name of your project.
Set the appropriate version, author, email, and project description.
List your project's dependencies under install_requires.
requirements.txt:
numpy
scikit-learn
# Add other dependencies as needed
In the requirements.txt file, list the dependencies required for your project, one per line.

Generating egg-info folder:

To generate the egg-info folder, you can use the following command in your project directory:
python setup.py egg_info
This command will generate the necessary metadata files for your project in an egg-info directory. The egg-info folder contains information about your project, such as its name, version, and dependencies. It's used by tools like pip to manage package installations.

Once you've created the setup.py and requirements.txt files and generated the egg-info folder, your project is ready for packaging and distribution. You can use these files to create distributable packages or install your project's dependencies using pip.
# In[ ]:





# H rite the logging function in loggerIpy and exception function in exceptionIpy file to be used for the
# project to track the progress when the ML project is run and to raise any exception when encounteredI
Certainly! Below are sample implementations for the logging and exception handling functions in the logger.py and exception.py files for your machine learning project.

logger.py - Implementing a logging function:
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler for logging
log_file = 'project.log'
file_handler = logging.FileHandler(log_file)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

def log_info(message):
    """
    Log an informational message.
    """
    logger.info(message)

def log_warning(message):
    """
    Log a warning message.
    """
    logger.warning(message)

def log_error(message):
    """
    Log an error message.
    """
    logger.error(message)

def log_exception(exception_message, exception_traceback):
    """
    Log an exception and its traceback.
    """
    logger.error(f'Exception: {exception_message}')
    logger.error(f'Traceback:\n{exception_traceback}')
In logger.py, we create a logger, configure it to log to a file, and define functions for logging information, warnings, errors, and exceptions.

exception.py - Implementing an exception handling function:
class MLProjectException(Exception):
    """
    Custom exception class for ML project.
    """
    pass

def handle_exception(exception_message):
    """
    Handle an exception by logging it and raising MLProjectException.
    """
    import traceback
    from .logger import log_exception

    exception_traceback = traceback.format_exc()
    log_exception(exception_message, exception_traceback)
    raise MLProjectException(exception_message)
In exception.py, we define a custom exception class MLProjectException to be used for handling exceptions specific to your ML project. We also define a handle_exception function that logs an exception message and raises MLProjectException with the same message.

You can now use these logging and exception handling functions in your project. For example:
# In your ML code
from your_ml_project.logger import log_info, log_warning, log_error
from your_ml_project.exception import handle_exception

try:
    # Your ML code here
    log_info("Model training started.")
    # ...
    if something_went_wrong:
        raise ValueError("An error occurred.")
except Exception as e:
    handle_exception(str(e))
This way, you can log progress and handle exceptions in your ML project effectively.
# In[ ]:





# In the notebook folder create a jupyter notebook inside it and do the following with the dataset:
# - Exploratory Data Analysis
# - Feature Engineering
# - Model Training
# - Selection of best model using metric
Creating and executing a Jupyter Notebook to perform Exploratory Data Analysis (EDA), Feature Engineering, Model Training, and Model Selection using a dataset requires a step-by-step interactive process. Below, I'll outline how you can create and structure your Jupyter Notebook for this task. Please note that this is a high-level overview, and you'll need to write and execute code within the notebook for each step.

Step 1: Create a Jupyter Notebook

Open a terminal and navigate to your project directory:
cd your_ml_project/notebook
Create a Jupyter Notebook:
jupyter notebook
This will open Jupyter Notebook in your web browser. Create a new Python 3 notebook and save it with a relevant name, such as "ML_Project_EDA_Feature_Engineering_Model_Training.ipynb."

Step 2: Load the Dataset

In your Jupyter Notebook, start by loading the Breast Cancer dataset (or any other dataset you are working with) using scikit-learn or any other data loading method you prefer.
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
Step 3: Exploratory Data Analysis (EDA)

Perform EDA to gain insights into your dataset. Some common EDA tasks include:

Data summary statistics.
Data visualization (histograms, scatter plots, etc.).
Handling missing values.
Identifying and handling outliers.
You can use libraries like pandas, matplotlib, and seaborn for EDA. Here's a sample EDA code block:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame from the dataset
df = pd.DataFrame(data.data, columns=data.feature_names)

# Summary statistics
summary = df.describe()

# Data visualization
sns.pairplot(df[['mean radius', 'mean texture', 'mean area', 'mean smoothness', 'target']], hue='target')
plt.show()
Step 4: Feature Engineering

In this step, you can perform feature engineering tasks such as feature selection, dimensionality reduction, and creating new features based on domain knowledge. This part heavily depends on your dataset and problem domain.

Step 5: Model Training

Train machine learning models using the processed data. You can use scikit-learn or other machine learning libraries for this task. Here's an example of training a simple classifier:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
Step 6: Model Selection using Metrics

Evaluate and compare multiple models using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score, ROC-AUC, etc.). You can create a table or plot to compare different models' performance.

This is a high-level overview of the steps you can take within your Jupyter Notebook. You'll need to write code and documentation within the notebook to perform each of these tasks in detail. Jupyter Notebook is a powerful tool for interactive data analysis and model development.
# In[ ]:





# write a separate python program in import_data.py file to load the mentioned dataset from sklearn.load_dataset.load_breast_cancer to your MongoDB
To load the Breast Cancer dataset from scikit-learn and save it to MongoDB using Python, you can create a separate script in your import_data.py file. You'll need to use the pymongo library to interact with MongoDB. Make sure you have pymongo installed. You can install it using pip if it's not already installed:
pip install pymongo
Here's a sample script to load the dataset and insert it into a MongoDB database:
from sklearn.datasets import load_breast_cancer
import pymongo

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# MongoDB connection information
mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
db_name = "your_db_name"  # Replace with your database name
collection_name = "breast_cancer_data"  # Replace with your desired collection name

# Establish a connection to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Prepare the data for insertion into MongoDB
documents = []
for i in range(len(X)):
    document = {
        "features": X[i].tolist(),
        "target": int(y[i])
    }
    documents.append(document)

# Insert the data into MongoDB
insert_result = collection.insert_many(documents)

# Print the number of documents inserted
print(f"Inserted {len(insert_result.inserted_ids)} documents into MongoDB collection: {collection_name}")
Replace "mongodb://localhost:27017/" with your MongoDB URI, "your_db_name" with your desired database name, and "breast_cancer_data" with your preferred collection name.

After running this script, the Breast Cancer dataset will be loaded from scikit-learn and inserted into the specified MongoDB database and collection.
# In[ ]:





# In data_ingestion.py write a program to load the same dataset from the MongoDB to your system in DataFrame format.
To load data from MongoDB to a DataFrame format using Python, you can create a program in your data_ingestion.py file. You'll need to use the pymongo library to interact with MongoDB and the pandas library to work with DataFrames. Make sure you have both libraries installed:

pip install pymongo pandas
Here's a sample script to load data from MongoDB into a DataFrame:

import pymongo
import pandas as pd

# MongoDB connection information
mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
db_name = "your_db_name"  # Replace with your database name
collection_name = "breast_cancer_data"  # Replace with your collection name

# Establish a connection to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Query MongoDB to retrieve data
cursor = collection.find()

# Convert data to a list of dictionaries
data_list = list(cursor)

# Create a DataFrame from the data
df = pd.DataFrame(data_list)

# Optionally, drop the "_id" field added by MongoDB
df = df.drop("_id", axis=1)

# Now, df contains your data in DataFrame format
Replace "mongodb://localhost:27017/", "your_db_name", and "breast_cancer_data" with your MongoDB connection information and the appropriate database and collection names.

After running this script, the data from MongoDB will be loaded into the DataFrame df, and you can work with it as needed for further analysis or processing.
# In[ ]:





# 10. Do the necessary feature engineering part in data_transformation.py.
Feature engineering is a crucial step in the machine learning pipeline, and it involves creating new features, transforming existing ones, and preparing the data for model training. In your data_transformation.py file, you can perform feature engineering on your dataset. Below is an example of how you can perform feature engineering tasks on the Breast Cancer dataset using pandas:

import pandas as pd

# Assuming you have already loaded your data into a DataFrame named 'df'

# Feature Scaling (if needed, to normalize/standardize your data)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('target', axis=1)), columns=df.columns[:-1])

# Feature Selection (you can choose a method or criteria for feature selection)
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=10)  # Select the top 10 features based on chi-squared test
X_new = selector.fit_transform(df_scaled, df['target'])

# Creating New Features (based on domain knowledge or feature engineering techniques)
# For example, you can create interaction features, polynomial features, etc.

# Removing Unnecessary Features
# If certain features are irrelevant, you can drop them from the DataFrame

# Encoding Categorical Variables (if applicable)
# Convert categorical variables to numerical format (e.g., one-hot encoding)

# Handling Missing Values (if applicable)
# You can fill missing values or drop rows/columns with missing data

# Data Transformation (if needed, apply transformations such as log, square root, etc.)

# Handling Imbalanced Data (if applicable)
# Implement techniques like oversampling or undersampling for class imbalance

# After feature engineering, save the transformed DataFrame back to a file or use it for model training
df_transformed = pd.concat([pd.DataFrame(X_new), df['target']], axis=1)
df_transformed.to_csv('transformed_data.csv', index=False)
In this example, various feature engineering tasks are performed, including feature scaling, feature selection, creating new features, and encoding categorical variables. Depending on your dataset and problem, you may need to perform specific feature engineering tasks or implement custom transformations.

Make sure to customize the feature engineering steps according to your dataset and the goals of your machine learning project.
# In[ ]:





# 11. Create the Machine Learning Model in Model_trainer.py.
To create a machine learning model in model_trainer.py, you can use scikit-learn or any other machine learning library based on your project's requirements. Below is an example of how to train a simple classifier, such as Logistic Regression, on the Breast Cancer dataset using scikit-learn:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed data (assuming it's saved as 'transformed_data.csv')
df = pd.read_csv('transformed_data.csv')

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Generate a classification report for more detailed evaluation
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Optionally, save the trained model for future use
from joblib import dump
dump(model, 'trained_model.joblib')
In this example, we load the preprocessed data, split it into training and testing sets, train a Logistic Regression model, and evaluate its performance using accuracy and a classification report. You can replace the model with any other machine learning algorithm of your choice, such as Random Forest, Support Vector Machine, or Neural Networks, depending on your project requirements.

Make sure to customize the code according to your dataset and the specific machine learning model you want to train.
# In[ ]:





# 12. Use Flask to deplay your project.
Deploying a machine learning project using Flask involves creating a web application that exposes your model through an API. Here are the steps to deploy your project using Flask:

Step 1: Install Flask

If you haven't already installed Flask, you can do so using pip:

pip install Flask
Step 2: Create the Flask App

Create a new Python file (e.g., app.py) to define your Flask application. Here's a basic example:

from flask import Flask, request, jsonify
import pandas as pd
from joblib import load  # If you saved your model using joblib

app = Flask(__name__)

# Load your trained model
model = load('trained_model.joblib')  # Replace with the path to your saved model

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()

        # Prepare the data for prediction (assuming it's in the right format)
        features = pd.DataFrame(data).values

        # Make predictions
        predictions = model.predict(features)

        # Convert predictions to a JSON response
        response = {'predictions': predictions.tolist()}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
This code sets up a Flask application with an endpoint (/predict) that accepts POST requests containing data to make predictions.

Step 3: Deploy the Flask App

You can deploy your Flask app in different ways depending on your hosting preferences. Here are a few options:

Local Development: Run the Flask app on your local machine by executing python app.py. It will be accessible at http://127.0.0.1:5000/.

Heroku: Deploy your Flask app to Heroku, a popular platform-as-a-service (PaaS) provider. You'll need to create a Procfile and configure a Heroku application. Refer to Heroku's documentation for deployment steps.

AWS, Azure, GCP: Deploy your Flask app to cloud platforms like AWS, Azure, or Google Cloud. Each platform provides its own deployment options and documentation.

Docker: Containerize your Flask app using Docker and deploy it to container orchestration platforms like Kubernetes.

Other Hosting Services: You can use hosting services like PythonAnywhere, DigitalOcean, or VPS providers to deploy your Flask app.

The choice of deployment platform depends on your project's requirements and your familiarity with the hosting service.

Step 4: Test Your Deployed API

After deploying your Flask app, you can test your machine learning model by sending POST requests to the /predict endpoint. Use tools like curl or libraries like requests in Python to send JSON data to the API and receive predictions in response.

For example, you can test your API locally with curl:

curl -X POST -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0}' http://127.0.0.1:5000/predict
Replace "feature1", "feature2", and the data format with your actual features and data.

This is a high-level overview of deploying a Flask-based machine learning project. Depending on your project's complexity and hosting requirements, you may need to configure additional settings, handle security, and consider scalability issues.
# In[ ]:





# #  <P style="color:green"> THAT'S ALL, THANK YOU    </p>
