#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> SUPPORT VECTOR MACHINES-1   </p>

# Q1. What is the mathematical formula for a linear SVM?
The mathematical formula for a linear Support Vector Machine (SVM) can be expressed as follows:

Given a dataset of labeled points {xi, yi}, where xi represents the input data points and yi represents their corresponding class labels (either +1 or -1 for a binary classification problem), the goal of a linear SVM is to find a hyperplane that maximally separates the data points of different classes.

1. Linear SVM Model:
   The decision function for a linear SVM is defined as:

   f(x) = wx + b

   Here:
   - f(x) is the decision function that assigns an input data point x to one of the two classes based on the sign of f(x).
   - w is the weight vector, which represents the coefficients of the hyperplane.
   - x is the input data point.
   - b is the bias term.

2. Objective Function:
   The objective of a linear SVM is to find the optimal weight vector w and bias term b that maximize the margin between the two classes while minimizing the classification error. This can be formulated as an optimization problem:

   Minimize: (1/2) ||w||^2
   Subject to: yi(wxi + b) ≥ 1 for all i

   Here:
   - ||w||^2 represents the L2 norm (Euclidean norm) of the weight vector w.
   - yi(wxi + b) is the margin for each data point xi.

3. Constraints:
   The constraints ensure that all data points are correctly classified and that the margin is at least 1 for all data points.

The SVM aims to find the optimal w and b that satisfy these constraints while maximizing the margin, which is the distance between the hyperplane and the closest data points of each class.

The optimization problem can be solved using various techniques, such as quadratic programming or gradient descent, to find the values of w and b that define the separating hyperplane for the given dataset.
# In[ ]:





# Q2. What is the objective function of a linear SVM?
The objective function of a linear Support Vector Machine (SVM) is a mathematical expression that defines the optimization goal when training a linear SVM. The objective of a linear SVM is to find the parameters (weight vector, w, and bias term, b) that define a hyperplane which maximally separates the data points of different classes while minimizing the classification error. The objective function for a linear SVM is typically expressed as follows:

Minimize: (1/2) ||w||^2

Subject to: yi(wxi + b) ≥ 1 for all i

Here's a breakdown of the components of the objective function:

1. (1/2) ||w||^2: This part of the objective function is a regularization term, where ||w|| represents the L2 norm (Euclidean norm) of the weight vector w. The goal is to minimize this term, which effectively encourages finding a weight vector with small magnitudes. This regularization term helps prevent overfitting and leads to a simpler decision boundary.

2. yi(wxi + b): This part of the objective function represents the margin for each data point xi. The margin is the distance between a data point and the separating hyperplane, scaled by the class label yi (which is either +1 or -1 for a binary classification problem). The objective is to ensure that all data points are correctly classified with a margin of at least 1. If this inequality is satisfied for all data points, it means that the data points are correctly classified and lie on the correct side of the hyperplane.

The overall objective is to find the values of w and b that minimize the regularization term (1/2) ||w||^2 while satisfying the constraint that yi(wxi + b) ≥ 1 for all data points. This effectively finds the hyperplane that maximizes the margin between the two classes and correctly classifies the data.

The optimization problem can be solved using various techniques, such as quadratic programming or gradient descent, to find the optimal values of w and b that achieve this objective.
# In[ ]:





# Q3. What is the kernel trick in SVM?
The kernel trick is a fundamental concept in Support Vector Machines (SVMs) that allows SVMs to efficiently handle non-linearly separable data by implicitly mapping the data into a higher-dimensional feature space. It's a mathematical technique that enables SVMs to find complex decision boundaries in this higher-dimensional space while still performing computations in the original input space. The kernel trick is particularly useful for solving non-linear classification problems.

In a linear SVM, you find a hyperplane that best separates data points in their original feature space. However, when dealing with data that is not linearly separable, trying to find a linear boundary may not be effective. This is where the kernel trick comes into play:

1. **Kernel Functions:** Instead of explicitly mapping data points to the higher-dimensional space, which can be computationally expensive or even impossible for very high dimensions, SVMs use kernel functions. A kernel function is a mathematical function that computes the dot product between the mapped (implicitly) feature vectors in the higher-dimensional space without actually calculating the mapping explicitly.

2. **Types of Kernels:** Commonly used kernel functions include:
   - **Linear Kernel (No Mapping):** K(x, y) = x * y (the dot product in the original space).
   - **Polynomial Kernel:** K(x, y) = (αx * y + c)^d, where α, c, and d are kernel parameters.
   - **Radial Basis Function (RBF) Kernel:** K(x, y) = exp(-γ * ||x - y||^2), where γ is a kernel parameter.
   - **Sigmoid Kernel:** K(x, y) = tanh(αx * y + c), where α and c are kernel parameters.

3. **Advantages:** The kernel trick allows SVMs to implicitly operate in a higher-dimensional space where data may become linearly separable even when it was not in the original space. This flexibility enables SVMs to model complex decision boundaries and capture intricate patterns in the data.

4. **Computational Efficiency:** The key advantage of the kernel trick is that it doesn't require explicitly computing the transformation of data into the higher-dimensional space. Instead, it computes the dot product directly using the kernel function, which can be much more efficient than performing the actual mapping.

By choosing an appropriate kernel function and tuning its parameters, you can adapt SVMs to handle various non-linear classification tasks effectively. The choice of kernel function depends on the specific problem and the characteristics of the data you are working with.
# In[ ]:





# Q4. What is the role of support vectors in SVM Explain with example
Support vectors play a crucial role in Support Vector Machines (SVMs). They are the data points from the training dataset that are closest to the decision boundary (hyperplane) and have a direct influence on the construction of the hyperplane. Support vectors are critical because they help determine the position and orientation of the hyperplane, and they define the margin of the SVM. Here's an explanation with an example:

Suppose you have a binary classification problem with two classes, A and B, and your goal is to find a decision boundary that separates the two classes as effectively as possible.

1. **Collecting Data:**
   You collect a dataset with labeled points from both classes. Each data point consists of features and a class label (A or B).

2. **Training the SVM:**
   You use the SVM algorithm to train a classifier on this dataset. The SVM aims to find a hyperplane that best separates the data points into two classes. The decision boundary is determined by the parameters w (weight vector) and b (bias term) of the hyperplane.

3. **Support Vectors:**
   During the training process, the SVM identifies a subset of data points from the training set that are the "support vectors." These support vectors are the data points that are closest to the decision boundary or lie on the margin of the hyperplane. They are typically the most challenging or informative points in terms of separating the two classes.

4. **Margin:**
   The margin is the distance between the hyperplane and the nearest support vectors. The SVM's objective is to maximize this margin while ensuring that all support vectors are correctly classified. The margin is crucial because it represents the level of confidence in the classification. A wider margin indicates a more robust separation between classes.

5. **Influence on Hyperplane:**
   Support vectors directly influence the position and orientation of the hyperplane. The SVM ensures that the margin is maximized while still correctly classifying all support vectors. This means that any change in the position or orientation of the hyperplane would affect the support vectors and the margin.

Example:
Imagine a simple 2D classification problem where you have two classes of data points: red circles and blue squares. You want to separate them using an SVM with a linear kernel. In this case, the support vectors would be the data points that are nearest to the decision boundary (the hyperplane).

Here, let's say that the two red circles and two blue squares closest to the decision boundary become the support vectors. The SVM will position the decision boundary in such a way that it maximizes the margin between the two classes while ensuring that these four support vectors are correctly classified.

In summary, support vectors are the critical data points that dictate the position and orientation of the decision boundary in an SVM, and they play a pivotal role in defining the margin and the classifier's overall performance.
# In[ ]:





# Q5. Illustrate with examples and graphs of Hyperplane, Marginal plane, Soft margin and Hard margin in
# SVM?
To illustrate the concepts of hyperplane, margin, soft margin, and hard margin in Support Vector Machines (SVM), let's use a simple 2D example with two classes, A and B, and visualize these concepts graphically.

Data Points:
Suppose we have two classes, A (red circles) and B (blue squares), as shown in the following graph:
          A
        ● ●
        ● ●
          B
        ■ ■
        ■ ■
Now, let's explore each concept with visual examples:

1. Hyperplane:
The hyperplane is the decision boundary that separates the two classes. In a 2D space, it's a straight line. In this example, let's assume the hyperplane is a straight line that separates the two classes like this:

          A
        ● ●
        ● ●
   hyperplane
        ■ ■
        ■ ■
          B
2. Margin:
The margin is the region between the hyperplane and the nearest data points from each class. In a hard-margin SVM, the margin is maximized. In this example, let's consider a hard-margin SVM where the margin is maximized:

          A
        ● ●
      / ● ●
     /  |  |
margin |  |  |
     \  |  |
      \ ■ ■
        ■ ■
          B
The margin is the space between the dashed lines on each side of the hyperplane. It represents the region where we are most confident in the classification.

3. Soft Margin:
In a soft-margin SVM, some data points may be allowed to be inside the margin or even on the wrong side of the hyperplane if the data is not linearly separable. This is to achieve a balance between maximizing the margin and allowing some misclassification. Here's an example:

          A
        ● ●
      / ● ●
     /  |  |
margin |  |  |
     \  |  |
      \   ■ ■
        ■ ■
          B
In this soft-margin example, a blue square (B) is inside the margin. This is allowed in a soft-margin SVM to accommodate some level of misclassification.

4. Hard Margin:
In a hard-margin SVM, there is no tolerance for misclassification. The margin is maximized, and all data points are correctly classified. Here's an example:

          A
        ● ●
        ● ●
     hyperplane
        ■ ■
        ■ ■
          B
In this hard-margin example, all data points are correctly classified, and there is no room for any point to be inside the margin.

Keep in mind that in practice, whether you use a hard-margin or soft-margin SVM depends on the data and the problem you're solving. A hard-margin SVM is suitable when the data is perfectly separable, while a soft-margin SVM is more flexible and can handle noisy or overlapping data. The choice of margin type and the position of the hyperplane depends on the specific problem you are addressing.
# In[ ]:





# Q6. SVM Implementation through Iris dataset.
Implementing a Support Vector Machine (SVM) using the Iris dataset is a common machine learning exercise. The Iris dataset is a well-known dataset for classification tasks, where the goal is to classify iris flowers into one of three species based on features like sepal length, sepal width, petal length, and petal width. Here's a step-by-step implementation of an SVM using Python and the Iris dataset:

Import Necessary Libraries:

You'll need libraries such as scikit-learn, NumPy, and Matplotlib for this task. Make sure you have them installed.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
Load and Prepare the Data:

Load the Iris dataset, and split it into features (X) and target labels (y).
iris = datasets.load_iris()
X = iris.data
y = iris.target
Split Data into Training and Testing Sets:

Split the data into training and testing sets to evaluate the SVM's performance.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Feature Scaling:

Standardize the features to have zero mean and unit variance, which can help SVMs perform better.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Create and Train the SVM Model:

Create an SVM classifier and fit it to the training data.

svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)
You can change the kernel and hyperparameters according to your requirements. In this example, we use a linear kernel.

Make Predictions:

Use the trained SVM model to make predictions on the test data.

y_pred = svm.predict(X_test)
Evaluate the Model:

Assess the model's performance by computing accuracy and other metrics.

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
This code will print out the accuracy, confusion matrix, and classification report, giving you a comprehensive view of how well the SVM model is performing on the Iris dataset.

That's a basic implementation of an SVM using the Iris dataset. You can further fine-tune the model by adjusting hyperparameters and experimenting with different kernel functions to achieve better classification results if needed.
# In[ ]:





# ~ Load the iris dataset from the scikit-learn library and split it into a training set and a testing setl
To load the Iris dataset from scikit-learn and split it into a training set and a testing set, you can follow these steps u
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into a training set and a testing set (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the dimensions of the resulting sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
In this code:

We import the datasets module from scikit-learn to access the Iris dataset.
We load the dataset into the variables X (features) and y (target labels).
We use the train_test_split function to split the dataset into training and testing sets. We specify test_size=0.2 to split the data into an 80% training set and a 20% testing set. The random_state parameter ensures reproducibility.
Finally, we print out the dimensions of the resulting sets to verify the split.
This code will give you two sets of data (X_train and y_train for training, and X_test and y_test for testing) that you can use for training and evaluating machine learning models.
# In[ ]:





# ~ Train a linear SVM classifier on the training set and predict the labels for the testing setl

To train a linear Support Vector Machine (SVM) classifier on the training set and predict the labels for the testing set using scikit-learn, you can follow these steps:

from sklearn.svm import SVC

# Create a linear SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier on the training set
svm_classifier.fit(X_train, y_train)

# Predict labels for the testing set
y_pred = svm_classifier.predict(X_test)

# Display the predicted labels
print("Predicted labels for the testing set:")
print(y_pred)
In this code:
We import the SVC (Support Vector Classifier) class from scikit-learn.
We create an instance of the SVC class with a linear kernel by specifying kernel='linear'.
We train the classifier on the training set using the fit method.
We use the trained classifier to predict labels for the testing set by calling the predict method on X_test.
Finally, we print the predicted labels (y_pred) for the testing set.
This code will train a linear SVM classifier on the training data and use it to make predictions on the testing data
# In[ ]:





# ~ Compute the accuracy of the model on the testing setl
To compute the accuracy of the SVM model on the testing set, you can use scikit-learn's accuracy_score function. Here's how you can calculate and display the accuracy:
from sklearn.metrics import accuracy_score

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy
print(f"Accuracy on the testing set: {accuracy:.2f}")
In this code:

We import the accuracy_score function from sklearn.metrics.
We use the accuracy_score function to compute the accuracy by comparing the true labels (y_test) with the predicted labels (y_pred).
Finally, we print the accuracy, usually as a percentage (e.g., 0.95 for 95% accuracy).
This code will calculate and display the accuracy of the SVM model on the testing set.
# In[ ]:





# ~ Plot the decision boundaries of the trained model using two of the featuresl
To plot the decision boundaries of the trained SVM model using two of the features, you can follow these steps. First, let's choose two features for visualization, and then we'll create a contour plot to display the decision boundaries:

import numpy as np
import matplotlib.pyplot as plt

# Select two features (e.g., sepal length and sepal width)
feature1_index = 0  # Sepal length
feature2_index = 1  # Sepal width

# Extract the selected features from the training data
X_train_subset = X_train[:, [feature1_index, feature2_index]]

# Define a mesh grid to create a contour plot
x_min, x_max = X_train_subset[:, 0].min() - 1, X_train_subset[:, 0].max() + 1
y_min, y_max = X_train_subset[:, 1].min() - 1, X_train_subset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Use the trained SVM model to make predictions on the mesh grid
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a contour plot to visualize the decision boundaries
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
plt.scatter(X_train_subset[:, 0], X_train_subset[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel(iris.feature_names[feature1_index])
plt.ylabel(iris.feature_names[feature2_index])
plt.title("Decision Boundaries of the Trained SVM Model")
plt.show()
In this code:
We choose two features (sepal length and sepal width) for visualization by specifying their indices (feature1_index and feature2_index).
We extract these two features from the training data to create X_train_subset.
We define a mesh grid (xx and yy) that covers the feature space to create a contour plot.
We use the trained SVM model to make predictions on the mesh grid points, which defines the decision boundaries.
Finally, we create a contour plot overlaid with the training data points to visualize the decision boundaries.
This code will generate a contour plot showing the decision boundaries of the trained SVM model using the selected features.
# In[ ]:





# ~ Try different values of the regularisation parameter C and see how it affects the performance of
# the model.
To observe how different values of the regularization parameter C affect the performance of the SVM model, you can train and evaluate the model with various C values and compare the results. Lower values of C lead to stronger regularization, which may result in a simpler model but might underfit the data. Higher values of C lead to weaker regularization, which can result in a more complex model that may overfit the data. Here's how you can experiment with different C values and evaluate the model:

from sklearn.metrics import accuracy_score

# Define a list of different C values to try
C_values = [0.01, 0.1, 1, 10, 100]

# Initialize lists to store accuracy scores for each C value
accuracy_scores = []

# Loop through each C value and evaluate the model
for C in C_values:
    # Create a linear SVM classifier with the current C value
    svm_classifier = SVC(kernel='linear', C=C, random_state=42)
    
    # Train the classifier on the training set
    svm_classifier.fit(X_train, y_train)
    
    # Predict labels for the testing set
    y_pred = svm_classifier.predict(X_test)
    
    # Compute the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    print(f"C = {C}: Accuracy on the testing set: {accuracy:.2f}")

# Plot the accuracy scores for different C values
plt.figure(figsize=(8, 6))
plt.plot(C_values, accuracy_scores, marker='o')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Regularization Parameter (C) for SVM')
plt.xscale('log')
plt.grid(True)
plt.show()
In this code:
We define a list of different C values to experiment with (e.g., [0.01, 0.1, 1, 10, 100]).
We loop through each C value, create a linear SVM classifier with that C value, train the model, make predictions on the testing set, and compute the accuracy.
The accuracy scores for each C value are stored in the accuracy_scores list.
We also plot a graph to visualize how the accuracy changes with different C values.
Running this code will give you insights into how different regularization strengths (controlled by C) impact the performance of the SVM model. You can observe the trade-off between underfitting and overfitting as you adjust the C parameter.
# In[ ]:





# 
# #  <P style="color:GREEN"> Thank You ,That's All </p>
