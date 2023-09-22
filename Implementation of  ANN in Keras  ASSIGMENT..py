#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# # <P style="color:purple"> Implementation of ANN in Keras </p>

# Q1. Install and load the latest versions of TensorFlow and Keras. Print their versions.
# Install the latest version of TensorFlow
!pip install tensorflow

# Install the latest version of Keras (usually included with TensorFlow)
!pip install keras

# Import TensorFlow and Keras and print their versions
import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
Running this code in your Python environment will install the latest versions of TensorFlow and Keras and then print their versions to the console.
Make sure you have Python and pip installed on your system before running this code.
# In[ ]:





# Q2. Load the Wine Quality dataset and explore its dimensions.
# Dataset link:
# https://www.kaggle.com/datasets/nareshbhat/wine-quality-binary-classification

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('C:\\Users\\MOHD. ARSHAD\\Downloads\\archive (1)\\\wine.csv')


# In[4]:


data


# In[ ]:





# Q3. Check for null values, identify categorical variables, and encode them.

# In[5]:


from sklearn.preprocessing import LabelEncoder

Step 1: Check for null values
To check for null values in the DataFrame, you can use the isnull() function along with the sum() function to count the number of null values in each column.
null_values = data.isnull().sum()
print(null_values)
This will display the number of null values in each column of your dataset.
Step 2: Identify categorical variables
You can identify categorical variables by checking the data type of each column in your DataFrame. Categorical variables are typically of type 'object' or 'category' in pandas.
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
print(categorical_columns)
This will give you a list of column names that contain categorical variables.
Step 3: Encode categorical variables
To encode categorical variables, you can use Label Encoding or One-Hot Encoding, depending on the nature of your data. Here, I'll demonstrate Label Encoding.
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
This code will encode the categorical variables in your DataFrame using Label Encoding. If you prefer One-Hot Encoding, you can use the pd.get_dummies() function.
Remember to save the preprocessed data if needed:
data.to_csv('preprocessed_dataset.csv', index=False)
This saves the preprocessed data to a new CSV file named 'preprocessed_dataset.csv'. You can adjust the encoding method and file format according to your specific requirements.
# In[ ]:





# Q4. Separate the features and target variables from the dataframe.
To separate the features (independent variables) and the target variable (dependent variable) from a DataFrame in Python, you can use the following steps:

Assuming your DataFrame is named data and the target variable is in a column named 'target':
# Separate the features (X) and the target (y)
X = data.drop(columns=['target'])  # Replace 'target' with the actual column name of your target variable
y = data['target']

# Optionally, you can also convert y to a numpy array if needed:
# y = y.values
In this code:

X contains all the columns from the DataFrame data except for the one containing the target variable.
y contains only the column with the target variable.
Now, you have X as your feature matrix and y as your target variable, which are ready for use in machine learning models or any other analysis you intend to perform.
# Q5. Perform a train-test split and divide the data into training, validation, and test datasets.
Performing a train-test split and dividing your data into training, validation, and test datasets is crucial for machine learning model evaluation and tuning. Here's how you can do it using Python and the scikit-learn library:

python
Copy code
from sklearn.model_selection import train_test_split

# Split the data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optionally, you can print the shapes of the resulting sets to verify the split
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)
In the code above:

X is your feature matrix.
y is your target variable.
We first split the data into a training set (70% of the data) and a temporary set (X_temp and y_temp) containing the remaining 30%.
Then, we further split the temporary set into a validation set (15% of the original data) and a test set (15% of the original data).
You can adjust the test_size parameter to control the proportions of the split, and the random_state parameter ensures reproducibility by fixing the random seed for the split.

Now, you have three sets:

X_train and y_train: Training data and labels used to train your machine learning model.
X_val and y_val: Validation data and labels used for hyperparameter tuning and model selection.
X_test and y_test: Test data and labels used for evaluating the final model's performance.
Make sure to use the validation set for tuning your model's hyperparameters and the test set for assessing its generalization performance.
# In[ ]:




Q6. Scale the dataset using an appropriate scaling technique.Scaling a dataset is a common preprocessing step in machine learning to ensure that all features have the same scale or magnitude. This is important because some machine learning algorithms are sensitive to the scale of the input features. One common scaling technique is called "Standardization" or "Z-score normalization," which scales the data to have a mean of 0 and a standard deviation of 1. Here's how you can scale a dataset using Python and the popular library, scikit-learn:
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assuming you have your dataset in a variable 'X'
# X should be a 2D numpy array or pandas DataFrame

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)

# X_scaled now contains the scaled data
In the code above:
Import the necessary libraries, including NumPy for numerical operations and scikit-learn's StandardScaler for scaling.
Create an instance of the StandardScaler.
Use the fit_transform method to both compute the mean and standard deviation of the dataset and scale it. This method returns the scaled dataset, which you can store in a variable like X_scaled.
After scaling your dataset, it's important to use the scaled data for training your machine learning models to ensure consistent and stable performance.
# In[ ]:




Q7. Design and implement at least two hidden layers and an output layer for the binary categorical
variables.To design and implement a neural network with at least two hidden layers and an output layer for binary categorical variables using Keras (which is commonly used for deep learning tasks), you can follow this example. In this example, we'll create a sequential model with two hidden layers and an output layer for binary classification.
Assuming you have your dataset prepared with features (X) and binary categorical labels (y), you can build the neural network as follows:
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a sequential model
model = keras.Sequential()

# Add the first hidden layer with 32 units, ReLU activation, and input shape (assuming X has features)
model.add(keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))

# Add the second hidden layer with 16 units and ReLU activation
model.add(keras.layers.Dense(16, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation for binary classification
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
In this code:
We first split the dataset into training and testing sets using train_test_split from scikit-learn.
We create a sequential model using keras.Sequential().
We add the first hidden layer with 32 units and ReLU activation. The input_shape is set to match the number of features in your dataset.
We add the second hidden layer with 16 units and ReLU activation.
We add the output layer with 1 unit and sigmoid activation, which is suitable for binary classification tasks.
We compile the model with an 'adam' optimizer and 'binary_crossentropy' loss, which is common for binary classification problems.
We train the model on the training data using model.fit.
Finally, we evaluate the model's performance on the test data and print the test loss and accuracy.
You can adjust the number of units in the hidden layers, the activation functions, and other hyperparameters to fine-tune the model for your specific problem.
# In[ ]:





# Q8. Create a Sequential model in Keras and add the previously designed layers to it.
Certainly! To create a Sequential model in Keras and add the previously designed layers to it, you can follow this code example. We'll create a new Sequential model and then add the layers we designed earlier.
import tensorflow as tf
from tensorflow import keras
# Initialize a sequential model
model = keras.Sequential()
# Add the first hidden layer with 32 units, ReLU activation, and input shape (assuming X has features)
model.add(keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)))
# Add the second hidden layer with 16 units and ReLU activation
model.add(keras.layers.Dense(16, activation='relu'))
# Add the output layer with 1 unit and sigmoid activation for binary classification
model.add(keras.layers.Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Summary of the model architecture
model.summary()
In this code, we:
Initialize a new Sequential model using keras.Sequential().
Add the first hidden layer with 32 units, ReLU activation, and input shape. You should adjust the input_shape according to the number of features in your dataset.
Add the second hidden layer with 16 units and ReLU activation.
Add the output layer with 1 unit and sigmoid activation, which is appropriate for binary classification tasks.
Compile the model with an 'adam' optimizer and 'binary_crossentropy' loss, which are common choices for binary classification.
Display a summary of the model's architecture using model.summary(). This summary provides information about the number of parameters and layer shapes in your model.
You can now use this Sequential model for training and evaluation as shown in the previous response.
# In[ ]:





# Q9. Print the summary of the model architecture.
To print the summary of the model architecture in Keras, you can use the summary() method of the model object. Here's how you can do it:

model.summary()
You should include this line of code after you have defined and compiled your model, as shown in the previous responses. When you run this line, it will display a summary of the model architecture in your console or Jupyter Notebook, providing information about the number of layers, the number of trainable parameters, and the shapes of each layer's output.
Here's an example of what the model summary might look like:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 32)                416
_________________________________________________________________
dense_1 (Dense)              (None, 16)                528
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 17
=================================================================
Total params: 961
Trainable params: 961
Non-trainable params: 0
_________________________________________________________________
This summary provides valuable information about your model's architecture, including the number of parameters and the shape of each layer's output.
# In[ ]:





# Q10. Set the loss function(‘binary_crossentropy’), optimizer, and include the accuracy metric in the model.
Certainly! In Keras, you can set the loss function, optimizer, and include the accuracy metric while compiling the model. Here's how you can do it:
import tensorflow as tf
from tensorflow import keras

# Initialize a sequential model
model = keras.Sequential()

# Add the first hidden layer with 32 units, ReLU activation, and input shape (assuming X has features)
model.add(keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)))

# Add the second hidden layer with 16 units and ReLU activation
model.add(keras.layers.Dense(16, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation for binary classification
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()
In this code:
We initialize a Sequential model.
We add layers to the model.
We compile the model using the compile method.
We set the optimizer to 'adam', which is a popular choice for optimization.
We set the loss function to 'binary_crossentropy', which is appropriate for binary classification tasks.
We include the 'accuracy' metric in the list of metrics to be computed during training.
With these settings, the model will use binary cross-entropy as the loss function, the Adam optimizer for training, and it will track accuracy as one of the evaluation metrics during training.
# In[ ]:





# Q11. Compile the model with the specified loss function, optimizer, and metrics.
Certainly! To compile the model with the specified loss function, optimizer, and metrics, you can use the compile method in Keras as follows:

import tensorflow as tf
from tensorflow import keras

# Initialize a sequential model
model = keras.Sequential()

# Add the first hidden layer with 32 units, ReLU activation, and input shape (assuming X has features)
model.add(keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)))

# Add the second hidden layer with 16 units and ReLU activation
model.add(keras.layers.Dense(16, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation for binary classification
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile the model with specified settings
model.compile(optimizer='adam',             # Optimizer: Adam
              loss='binary_crossentropy',  # Loss function: Binary Cross-Entropy
              metrics=['accuracy'])         # Metrics: Accuracy
In this code:
We initialize a Sequential model.
We add layers to the model.
We use the compile method to specify:
The optimizer, set to 'adam,' which is a commonly used optimizer.
The loss function, set to 'binary_crossentropy,' which is suitable for binary classification problems.
The metrics to be tracked during training, which includes 'accuracy' for measuring the classification accuracy of the model.
After compiling the model, you can proceed with training and evaluating it using your dataset and the specified settings.
# In[ ]:





# Q12. Fit the model to the training data using appropriate batch size and number of epochs.

To fit the compiled Keras model to the training data, you'll need to specify the batch size and the number of epochs for training. The batch size determines how many samples are used in each update of the model's weights during training, and the number of epochs defines how many times the entire training dataset is processed. Here's an example of how to fit the model with appropriate values for batch size and epochs:
# Fit the model to the training data
history = model.fit(X_train,             # Training features
                    y_train,             # Training labels
                    batch_size=32,       # Batch size (adjust as needed)
                    epochs=10,           # Number of epochs (adjust as needed)
                    validation_data=(X_test, y_test))  # Validation data for monitoring

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
In this code:
X_train and y_train represent your training features and labels.
You can adjust the batch_size to a value that fits your dataset and memory constraints. Common choices include 32, 64, or 128.
You can adjust the epochs to specify how many times the model should iterate over the entire training dataset. The appropriate number of epochs depends on your specific problem; you may need to experiment with different values.
The validation_data argument allows you to specify a validation dataset to monitor the model's performance during training.
After fitting the model to the training data, you can evaluate its performance on the test data, as shown in the code, by using the model.evaluate method.
# In[ ]:





# Q13. Obtain the model's parameters (weights and biases).
You can obtain the model's parameters, including the weights and biases, using the get_weights() method in Keras. This method returns a list of numpy arrays containing the parameters of each layer in the model. Here's how you can do it:
# Get the model's parameters (weights and biases)
model_params = model.get_weights()

# Print the model's parameters for each layer
for i, layer_params in enumerate(model_params):
    print(f"Layer {i + 1} Parameters:")
    for param in layer_params:
        print(param.shape)  # Print the shape of weights and biases
    print()
In this code:
model.get_weights() returns a list where each element corresponds to the weights and biases of a layer in the model.
The for loop iterates through the list of layer parameters and prints the shape of the weights and biases for each layer.
Keep in mind that the exact structure of the model_params list depends on your model's architecture and the number of layers you have. The list will contain the weights and biases for each layer in the order they were added to the model.
# In[ ]:





# Q14. Store the model's training history as a Pandas DataFrame.
To store the model's training history as a Pandas DataFrame, you can use the History object returned by the fit method when training the model. The History object contains information about the training and validation metrics at each epoch, which 
import pandas as pd

# Assuming you have already trained the model and stored the history object in 'history'
# The 'history' object is typically returned by the 'fit' method

# Create a Pandas DataFrame from the history object
history_df = pd.DataFrame(history.history)

# Display the first few rows of the DataFrame
print(history_df.head())

# You can now use 'history_df' for analysis and visualization
In this code:
history is the object returned when you train the model using the fit method.
We create a Pandas DataFrame called history_df using pd.DataFrame(history.history). This converts the training history, which includes metrics like loss and accuracy for each epoch, into a tabular format.
You can then use history_df for analysis, visualization, or further processing as needed. It will contain columns for training loss, training accuracy, validation loss, and validation accuracy for each epoch.
# In[ ]:





# Q15. Plot the training history (e.g., accuracy and loss) using suitable visualization techniques.
You can plot the training history, including accuracy and loss, using suitable visualization techniques. Matplotlib is a popular library for creating visualizations in Python. Here's an example of how to plot the training history of a Keras model:
import matplotlib.pyplot as plt

# Assuming you have already trained the model and stored the history object in 'history'
# The 'history' object is typically returned by the 'fit' method

# Extract training and validation accuracy and loss from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_accuracy) + 1)

# Create subplots for accuracy and loss
plt.figure(figsize=(12, 4))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
In this code:
We extract the training and validation accuracy, as well as the training and validation loss, from the history object.
We use Matplotlib to create two subplots: one for accuracy and one for loss.
In the accuracy subplot, we plot the training accuracy and validation accuracy over epochs.
In the loss subplot, we plot the training loss and validation loss over epochs.
We add labels, titles, and legends to the plots for clarity.
Finally, we display the plots using plt.show().
This code will produce two side-by-side plots showing the training and validation accuracy as well as the training and validation loss over the epochs, helping you visualize the model's performance during training.
# In[ ]:





# Q16. Evaluate the model's performance using the test dataset and report relevant metrics.
To evaluate the model's performance using the test dataset and report relevant metrics, you can use the evaluate method in Keras. This method computes various metrics, including loss and accuracy, on the test data. Here's how you can do it:
# Assuming you have already trained the model and compiled it

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Print the evaluation results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
In this code:
X_test and y_test represent your test features and labels.
The model.evaluate method computes the specified metrics (in this case, loss and accuracy) on the test data.
The computed loss and accuracy values are then printed to the console.
These metrics will give you an indication of how well your model performs on unseen data. The test accuracy is a common metric for classification tasks, while the test loss can provide insights into how well the model's predictions match the actual labels. Depending on your specific problem, you may want to consider other metrics as well, such as precision, recall, F1-score, etc., for a more comprehensive evaluation.
# #  <P style="color:GREEN"> Thank You ,That's All </p>
