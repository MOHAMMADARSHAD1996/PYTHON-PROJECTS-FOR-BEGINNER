#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# # <P style="color:purple"> Callback Functions </p>

# Q1. Install and load the latest versions of TensorFlow and Keras. Print their versions.
I can provide you with the code to install and load the latest versions of TensorFlow and Keras using Python. Please note that my knowledge is based on information available up to September 2021, and there may have been updates or changes since then. You can run the following code to install and load the latest versions of TensorFlow and Keras:
Install TensorFlow and Keras
!pip install tensorflow
!pip install keras
Import TensorFlow and Keras
import tensorflow as tf
import keras
Print the versions
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
Run this code in your Python environment to install and load the latest versions of TensorFlow and Keras, and it will print their respective versions. Keep in mind that the actual versions may be different from what was available up to September 2021.
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




Q6. Perform scaling on the dataset.Scaling is an important preprocessing step in machine learning, especially when working with algorithms that are sensitive to the scale of input features (e.g., gradient-based optimization algorithms like gradient descent). Two common methods for scaling features are Min-Max scaling and Standardization (Z-score scaling). Here, I'll demonstrate how to perform both types of scaling on a dataset using Python's scikit-learn library.

Assuming you have your feature matrix X ready:

1. Min-Max Scaling (Normalization):

Min-Max scaling scales the features to a specific range, typically [0, 1]. Here's how to do it:

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the feature matrix
X_scaled = scaler.fit_transform(X)
X_scaled now contains the scaled features.

2. Standardization (Z-score Scaling):

Standardization scales the features to have a mean of 0 and a standard deviation of 1. Here's how to do it:

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the feature matrix
X_scaled = scaler.fit_transform(X)
X_scaled now contains the standardized features.

Choose the scaling method based on your specific problem and the requirements of the machine learning algorithm you plan to use. Typically, Min-Max scaling is preferred when you want to bound your features to a specific range, while standardization is useful when you want to center the data around zero and have it follow a normal distribution.
# In[ ]:




Q7. Create at least 2 hidden layers and an output layer for the binary categorical variables.
To create a neural network model with at least two hidden layers and an output layer for binary categorical variables (i.e., binary classification), you can use a deep learning library like TensorFlow or Keras. Here's an example of how to create such a model using TensorFlow and Keras:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential()

# Add the input layer (assuming you have already preprocessed your data)
# You should specify the input shape based on the number of features in X_train.
input_shape = (X_train.shape[1],)  # Example: (number_of_features,)
model.add(layers.Input(shape=input_shape))

# Add at least two hidden layers
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=32, activation='relu'))

# Add the output layer for binary classification with a sigmoid activation function
model.add(layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary to view the architecture and the number of parameters
model.summary()
In this code:
We import the necessary TensorFlow and Keras modules.
We create a sequential model, which is a linear stack of layers.
The input layer is added implicitly when specifying the input_shape.
We add at least two hidden layers with the ReLU (Rectified Linear Unit) activation function, but you can customize the number of units and activation functions according to your problem.
For binary classification, we add an output layer with a single unit and a sigmoid activation function, which is common for binary classification tasks.
We compile the model using the Adam optimizer and binary cross-entropy loss, which is standard for binary classification. You can change the optimizer and loss function as needed.
Finally, we print the model summary to see the architecture and the number of parameters.
You should replace input_shape with the appropriate input shape based on the number of features in your dataset. Additionally, you can customize the architecture, activation functions, and other hyperparameters based on your specific problem and requirements.
# In[ ]:





# Q8. Create a Sequential model and add all the layers to it.
To create a Sequential model in Keras and add all the layers to it, you can follow these steps:

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add the input layer (assuming you have already preprocessed your data)
# You should specify the input_dim based on the number of features in your dataset.
input_dim = X_train.shape[1]  # Example: the number of features in X_train
model.add(Dense(units=64, activation='relu', input_dim=input_dim))

# Add at least two hidden layers
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))

# Add the output layer for binary classification with a sigmoid activation function
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary to view the architecture and the number of parameters
model.summary()
In this code:
We import the necessary TensorFlow and Keras modules.
We create a Sequential model.
The input layer is added explicitly with input_dim, which should be set to the number of features in your dataset.
We add at least two hidden layers with the ReLU (Rectified Linear Unit) activation function, but you can customize the number of units and activation functions as needed.
For binary classification, we add an output layer with a single unit and a sigmoid activation function, which is common for binary classification tasks.
We compile the model using the Adam optimizer and binary cross-entropy loss, which is standard for binary classification. You can change the optimizer and loss function as needed.
Finally, we print the model summary to see the architecture and the number of parameters.
Replace input_dim with the appropriate number based on the number of features in your dataset. Customize the architecture, activation functions, and other hyperparameters according to your specific problem and requirements.
# In[ ]:





# Q9. Implement a TensorBoard callback to visualize and monitor the model's training process.
Implementing a TensorBoard callback in Keras allows you to visualize and monitor the model's training process. TensorBoard is a powerful tool for tracking metrics, visualizing model architectures, and debugging. Here's how to implement a TensorBoard callback:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
# ...

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a TensorBoard callback
log_dir = "logs"  # Directory where TensorBoard logs will be stored
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with the TensorBoard callback
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

# To run TensorBoard, you can use the following command in your terminal:
# tensorboard --logdir=logs
In this code:
We import TensorFlow, Keras, and the necessary modules.
We create a Sequential model and add layers to it (input, hidden, and output layers). Be sure to define your model's architecture as needed.
We compile the model with the optimizer, loss function, and metrics of your choice.
We define a TensorBoard callback by specifying the log_dir, which is the directory where TensorBoard will store its logs. You can customize this directory path.
We train the model using the model.fit() method, and we include the TensorBoard callback in the callbacks parameter.
After training, you can run TensorBoard from your terminal using the tensorboard --logdir=logs command. Make sure to navigate to the directory containing your Python script to run this command.
TensorBoard will generate visualizations and logs that you can view in a web browser by accessing the URL provided in the terminal (usually something like http://localhost:6006/). You can monitor training metrics, visualizations of your model's architecture, and more.
# In[ ]:





# Q10. Use Early Stopping to prevent overfitting by monitoring a chosen metric and stopping the training if
# no improvement is observed.
Early stopping is a technique used to prevent overfitting by monitoring a chosen metric (usually validation loss or accuracy) and stopping the training process if no improvement is observed for a specified number of epochs. To implement early stopping in Keras, you can use the EarlyStopping callback. Here's how to do it:

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
# ...

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define an EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=3,           # Number of epochs with no improvement to wait
    restore_best_weights=True  # Restore the model weights to the best epoch
)

# Train the model with the EarlyStopping callback
history = model.fit(
    X_train, y_train,
    epochs=100,  # Maximum number of epochs
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_callback]
)
In this code:
We create a Sequential model and add layers to it (input, hidden, and output layers). Define your model architecture as needed.
We compile the model with the optimizer, loss function, and metrics of your choice.
We define an EarlyStopping callback with the following parameters:
monitor: The metric to monitor during training (in this case, we are monitoring validation loss).
patience: The number of epochs with no improvement in the monitored metric to wait before stopping training.
restore_best_weights: If set to True, the model's weights will be restored to the best weights obtained during training.
We train the model using the model.fit() method and include the early_stopping_callback in the callbacks parameter. The training process will stop early if no improvement in validation loss is observed for the specified number of epochs (patience).
The EarlyStopping callback will monitor the validation loss, and if it doesn't improve for the specified number of epochs (3 in this example), training will stop early. This helps prevent overfitting by stopping the training process when the model's performance on the validation set starts to degrade.
# In[ ]:





# Q11. Implement a ModelCheckpoint callback to save the best model based on a chosen metric during
# training.
The ModelCheckpoint callback in Keras allows you to save the best model based on a chosen metric during training. You can save the model with the lowest validation loss, highest validation accuracy, or any other metric of your choice. Here's how to implement the ModelCheckpoint callback:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
# ...

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5',  # Path to save the best model
    monitor='val_loss',        # Metric to monitor (e.g., validation loss)
    save_best_only=True,       # Save only the best model
    mode='min'                 # 'min' for loss, 'max' for accuracy, 'auto' to infer automatically
)

# Train the model with the ModelCheckpoint callback
history = model.fit(
    X_train, y_train,
    epochs=100,                # Maximum number of epochs
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback]
)
In this code:
We create a Sequential model and add layers to it (input, hidden, and output layers). Define your model architecture as needed.
We compile the model with the optimizer, loss function, and metrics of your choice.
We define a ModelCheckpoint callback with the following parameters:
filepath: The path where the best model will be saved (e.g., 'best_model.h5').
monitor: The metric to monitor during training (in this case, we are monitoring validation loss).
save_best_only: If set to True, the callback will save only the best model based on the monitored metric.
mode: 'min' for minimizing the metric (e.g., validation loss) or 'max' for maximizing the metric (e.g., validation accuracy). You can also use 'auto' to infer it automatically.
We train the model using the model.fit() method and include the checkpoint_callback in the callbacks parameter. The callback will save the best model during training based on the chosen metric (validation loss in this case).
After training, the best model will be saved to the specified file ('best_model.h5' in this example). You can later load this saved model for evaluation or inference using tf.keras.models.load_model('best_model.h5').
# In[ ]:





# Q12. Print the model summary.
To print the summary of a Keras model, you can use the summary() method of the model object. Here's how to print the summary of a Keras Sequential model:
Assuming you have created a model named model:
model.summary()
When you call model.summary(), it will display a summary of the model's architecture, including the number of trainable parameters in each layer, the layer type, and the output shape.
Here's a full example of how to create a simple Sequential model, compile it, and then print its summary:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
model.add(Dense(units=64, activation='relu', input_dim=10))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
Running the model.summary() command in this example will display the architecture and the number of parameters for each layer in the model. Adjust the architecture and layers based on your specific problem and requirements.
# In[ ]:





# Q13. Use binary cross-entropy as the loss function, Adam optimizer, and include the metric ['accuracy'].
Certainly, you can use binary cross-entropy as the loss function, the Adam optimizer, and include the metric 'accuracy' when compiling your Keras model. Here's how to do it:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
model.add(Dense(units=64, activation='relu', input_dim=10))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
In this code:
We create a Sequential model and add layers to it (input, hidden, and output layers).
When we compile the model using model.compile(), we specify the following:
optimizer='adam': We use the Adam optimizer.
loss='binary_crossentropy': We use binary cross-entropy as the loss function, which is suitable for binary classification problems.
metrics=['accuracy']: We include the 'accuracy' metric, which will be used to evaluate the model's performance during training and evaluation.
You can adjust the architecture and layers based on your specific problem, but the compilation settings you've specified will work for binary classification tasks.
# In[ ]:





# Q14. Compile the model with the specified loss function, optimizer, and metrics.
To compile a Keras model with the specified loss function, optimizer, and metrics, you can use the model.compile() method. Here's how to do it with binary cross-entropy loss, the Adam optimizer, and the 'accuracy' metric:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
model.add(Dense(units=64, activation='relu', input_dim=10))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with specified loss, optimizer, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
In this code:
We create a Sequential model and add layers to it (input, hidden, and output layers).
When we compile the model using model.compile(), we specify the following:
optimizer='adam': We use the Adam optimizer.
loss='binary_crossentropy': We use binary cross-entropy as the loss function, which is suitable for binary classification problems.
metrics=['accuracy']: We include the 'accuracy' metric, which will be used to evaluate the model's performance during training and evaluation.
These settings are now applied to the model, and you can proceed with training and evaluating it on your dataset.
# In[ ]:





# Q15. Fit the model to the data, incorporating the TensorBoard, Early Stopping, and ModelCheckpoint
# callbacks.
To fit the Keras model to the data while incorporating the TensorBoard, Early Stopping, and ModelCheckpoint callbacks, you can use the model.fit() method and pass the callbacks as a list in the callbacks parameter. Here's how to do it:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# Create a Sequential model
model = Sequential()

# Add layers to the model (input, hidden, output layers)
model.add(Dense(units=64, activation='relu', input_dim=10))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with specified loss, optimizer, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
log_dir = "logs"  # Directory where TensorBoard logs will be stored
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=5,           # Number of epochs with no improvement to wait
    restore_best_weights=True  # Restore the model weights to the best epoch
)

checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5',  # Path to save the best model
    monitor='val_loss',        # Metric to monitor (e.g., validation loss)
    save_best_only=True,       # Save only the best model
    mode='min'                 # 'min' for loss, 'max' for accuracy, 'auto' to infer automatically
)

# Fit the model to the data with the specified callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,                # Maximum number of epochs
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback]
)
In this code:
We create a Sequential model and add layers to it.
We compile the model with the specified loss function, optimizer, and metrics.
We define the callbacks, including TensorBoard, EarlyStopping, and ModelCheckpoint, with their respective configurations.
We use the model.fit() method to train the model on the training data, incorporating the callbacks by passing them in the callbacks parameter.
tensorboard_callback logs training information for TensorBoard.
early_stopping_callback stops training early if validation loss doesn't improve.
checkpoint_callback saves the best model during training based on validation loss.
You can adjust the settings of the callbacks and the number of training epochs based on your specific problem and requirements.
# In[ ]:





# Q16. Get the model's parameters.
To get the model's parameters (weights and biases) in Keras, you can use the get_weights() method on a layer or the entire model. Here's how you can retrieve the parameters for a specific layer and for the entire model:
For a Specific Layer:
To get the parameters of a specific layer, you can use the get_weights() method on that layer. For example, to get the parameters of the first dense layer in the model:
weights, biases = model.layers[0].get_weights()
In this code, model.layers[0] refers to the first layer in the model (assuming it's a dense layer), and get_weights() returns a tuple containing the layer's weights and biases.
For the Entire Model:
If you want to get all the parameters for the entire model, you can use the get_weights() method on the model itself. This will return a list of arrays containing the weights and biases for all layers in the model:
all_weights = model.get_weights()
In this case, all_weights is a list where each element is an array containing the weights and biases of a layer. The order of the elements in the list corresponds to the order of layers in the model.
You can then examine or manipulate these weights and biases as needed for further analysis or tasks.
# In[ ]:





# Q17. Store the model's training history as a Pandas DataFrame.
You can store the training history of a Keras model as a Pandas DataFrame by converting the history object returned by the model.fit() method. Here's how to do it:
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have trained the model and stored the history object as 'history'
history_dict = history.history

# Convert the history to a Pandas DataFrame
history_df = pd.DataFrame(history_dict)

# Display the first few rows of the DataFrame
print(history_df.head())

# Optionally, you can save the history DataFrame to a CSV file
history_df.to_csv('training_history.csv', index=False)

# Plot training and validation loss over epochs
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
In this code:
We assume you have trained the model and stored the training history in the history object.
We convert the history dictionary into a Pandas DataFrame using pd.DataFrame(history_dict).
Optionally, you can save the history DataFrame to a CSV file using to_csv().
We use Matplotlib to plot the training and validation loss over epochs to visualize the training progress.
The resulting history_df DataFrame will contain columns for training loss, validation loss, and any other metrics you specified during model compilation. You can access, analyze, or visualize this data as needed.
# In[ ]:





# Q18. Plot the model's training history.
To plot the training history of a Keras model, including training and validation loss and any other metrics, you can use the Matplotlib library. Here's how to do it:
import matplotlib.pyplot as plt

# Assuming you have trained the model and stored the history object as 'history'
history_dict = history.history

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))  # Optional: set the figure size
plt.plot(history_dict['loss'], label='Training Loss', marker='o')
plt.plot(history_dict['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation accuracy over epochs (if 'accuracy' is a recorded metric)
if 'accuracy' in history_dict:
    plt.figure(figsize=(10, 6))  # Optional: set the figure size
    plt.plot(history_dict['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
In this code:
We assume you have trained the model and stored the training history in the history object.
We retrieve the training and validation loss (and optionally accuracy) from the history object.
We use Matplotlib to create line plots for the training and validation loss. We also create a separate plot for accuracy if it's available in the training history.
You can adjust the figure size, markers, labels, titles, and other plot properties to customize the appearance of the plots.
Running this code will display line plots of the training and validation loss (and accuracy) over epochs, allowing you to visualize the training progress and identify potential overfitting or underfitting.
# In[ ]:





# Q19. Evaluate the model's performance using the test data.
To evaluate the performance of a trained Keras model using test data, you can use the model.evaluate() method. Here's how to do it:

# Assuming you have already trained the model and stored it as 'model'
# and you have test data 'X_test' and 'y_test'

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Print the test loss and accuracy
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')
In this code:
We assume that you have already trained the model and stored it as model.
You should also have your test data stored in X_test (feature data) and y_test (corresponding target labels).
We use the model.evaluate(X_test, y_test) method to evaluate the model's performance on the test data. This function returns the test loss and any specified metrics (e.g., accuracy) as a tuple.
We print the test loss and accuracy to the console.
This code will display the model's test loss and accuracy, allowing you to assess how well the model generalizes to unseen data.
# #  <P style="color:GREEN"> Thank You ,That's All </p>
