#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> TENSORFLOW FUNDAMENTALS </p>

# Objective: The objective of this assignment is to gain practical experience with fundamental operations
# in TensorFlow, including creating and manipulating matrices, performing arithmetic operations on
# tensors, and understanding the difference between TensorFlow constants and variables.

# # Part 1: Theoretical Questions
1.What are the different data structures used in Tensorflow?. Give some examples?TensorFlow, a popular deep learning framework, provides several data structures to handle and manipulate data efficiently. Some of the key data structures used in TensorFlow include:
Tensor:
A tensor is the fundamental data structure in TensorFlow. It is a multi-dimensional array, similar to a NumPy array, but with additional capabilities for GPU acceleration and distributed computing.
Examples:
Scalar (0-D tensor): tf.constant(5)
Vector (1-D tensor): tf.constant([1, 2, 3])
Matrix (2-D tensor): tf.constant([[1, 2], [3, 4]])
Higher-dimensional tensors: Tensors with more than two dimensions, e.g., for images or sequences.
tf.Variable:

tf.Variable is a special type of tensor that allows for mutable state. It's commonly used for model parameters that need to be updated during training.
Example:

weight_matrix = tf.Variable(initial_value=tf.random.normal(shape=(10, 10)))
tf.constant:

tf.constant is used to create tensors with constant values that cannot be changed once defined.
Example:

constant_tensor = tf.constant([1, 2, 3])
tf.placeholder (Deprecated in TensorFlow 2.x):

tf.placeholder was used in TensorFlow 1.x for creating placeholders for data that would be provided during the computation graph execution. In TensorFlow 2.x, eager execution and the use of Python native control structures have largely replaced placeholders.
SparseTensor:

SparseTensor is used to represent sparse data efficiently. It stores only non-zero values and their indices in a sparse tensor.
Example:
indices = tf.constant([[0, 1], [1, 2], [2, 3]])
values = tf.constant([3.0, 4.0, 5.0])
dense_shape = tf.constant([3, 4])
sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
RaggedTensor:

RaggedTensor is used to represent ragged or nested data structures, where the length of each element may vary.
Example:
ragged_tensor = tf.ragged.constant([[1, 2], [3, 4, 5], [6]])
StringTensor:

StringTensor is used to handle string data in TensorFlow.
Example:
string_tensor = tf.constant(["Hello", "TensorFlow"])
Queue and Dataset API:

TensorFlow provides a Queue API and a Dataset API for handling data pipelines efficiently. These APIs allow you to create input pipelines for your machine learning models.
Sparse Feature Columns (tf.feature_column):

These are used for handling categorical features in a structured way, especially in the context of TensorFlow's Estimator API for creating models.
Other Specialized Data Structures:

TensorFlow also offers specialized data structures like tf.TensorArray for dynamically growing arrays during graph execution and tf.SparseTensorValue for handling sparse tensors in session-based code.
These data structures play a crucial role in building and manipulating neural network models in TensorFlow, allowing you to work with a wide range of data types and structures efficiently. The choice of data structure depends on the nature of your data and the requirements of your machine learning or deep learning task.
# In[ ]:




2. How does the TensorFlow constant differ from a TensorFlow variable? Explain with an example?In TensorFlow, both constants and variables are used to represent and work with tensors, which are multi-dimensional arrays. However, there are key differences between TensorFlow constants and TensorFlow variables:
TensorFlow Constant:
A TensorFlow constant is a type of tensor whose value cannot be changed after it is defined. It remains constant throughout the execution of a TensorFlow program.
Constants are typically used for values that do not change during the computation, such as hyperparameters, fixed input data, or other static values.
Constants are created using the tf.constant function.
TensorFlow Variable:
A TensorFlow variable is a special type of tensor that allows its value to be changed during the execution of a TensorFlow program. Variables are used for model parameters that need to be updated and learned during training.
Variables are essential for building machine learning models because they store the learned weights and biases of the model.
Variables are created using the tf.Variable class.
Here's an example that illustrates the difference between TensorFlow constants and TensorFlow variables:

import tensorflow as tf

# Creating a TensorFlow constant
constant_tensor = tf.constant([1, 2, 3])

# Attempting to change the value of a constant will result in an error
try:
    constant_tensor.assign([4, 5, 6])  # This will raise an error
except Exception as e:
    print(f"Error: {e}")

# Creating a TensorFlow variable
variable_tensor = tf.Variable([1, 2, 3])

# Variables can be changed using assign method or by using operators
variable_tensor.assign([4, 5, 6])

# Printing the updated value of the variable
print("Updated Variable:", variable_tensor.numpy())
In this example:

constant_tensor is a TensorFlow constant with a fixed value of [1, 2, 3]. Attempting to change its value using the assign method will result in an error because constants are immutable.

variable_tensor is a TensorFlow variable initially set to [1, 2, 3]. Variables can be updated using the assign method or standard operators. In this case, we assign the value [4, 5, 6] to the variable, and the updated value is printed as [4, 5, 6]. Variables are mutable and are used for storing trainable parameters in machine learning models.
In summary, the key difference is that constants are immutable and their values cannot be changed once defined, while variables are mutable and allow their values to be updated during the execution of a TensorFlow program. Variables are typically used to store and update model parameters during training.
# In[ ]:




3. Describe the process of matrix addition, multiplication, and elementDwise operations in TensorFlow.In TensorFlow, you can perform various matrix and tensor operations, including addition, multiplication, and element-wise operations, using its computational graph-based approach. Here's a description of these operations in TensorFlow:
1. Matrix Addition:
Matrix addition is performed using the tf.add() function or the + operator in TensorFlow.
Example:
import tensorflow as tf

# Create two matrices
matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])

# Perform matrix addition
result = tf.add(matrix_a, matrix_b)
2. Matrix Multiplication:
Matrix multiplication can be performed using the tf.matmul() function or the @ operator in TensorFlow.
Example:
import tensorflow as tf

# Create two matrices
matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])

# Perform matrix multiplication
result = tf.matmul(matrix_a, matrix_b)
Note that for element-wise multiplication, you can simply use the * operator.

3. Element-wise Operations:

Element-wise operations are operations performed independently on each element of a tensor without regard to its shape.
Common element-wise operations include addition, subtraction, multiplication, and division.
You can perform element-wise operations using standard arithmetic operators (+, -, *, /) or TensorFlow functions like tf.add(), tf.subtract(), tf.multiply(), and tf.divide().
Example:
import tensorflow as tf

# Create two tensors
tensor_a = tf.constant([1, 2, 3])
tensor_b = tf.constant([4, 5, 6])

# Perform element-wise addition
result_add = tf.add(tensor_a, tensor_b)

# Perform element-wise multiplication
result_multiply = tf.multiply(tensor_a, tensor_b)
In summary, TensorFlow provides functions and operators to perform matrix addition, multiplication, and element-wise operations efficiently. These operations are essential for building and training deep learning models, as they form the core of many mathematical computations involved in neural networks. When performing these operations, make sure to pay attention to the shapes and dimensions of your tensors to ensure compatibility and correctness in your computations.
# In[ ]:





# # Part 2: Practical Implementation

# # Talk 1: Creating and Manipulating Matrices

# 1. Create a normal matrix A with dimensions 2x2, using TensorFlow's random_normal function. Display the values of matrix A.
You can create a 2x2 matrix with random values from a normal distribution using TensorFlow's tf.random.normal function. Here's how to create such a matrix and display its values:
import tensorflow as tf
# Create a 2x2 matrix with random values from a normal distribution
matrix_A = tf.random.normal(shape=(2, 2))
# Display the values of matrix A
print("Matrix A:")
print(matrix_A.numpy())
In this code:
tf.random.normal generates a random 2x2 matrix with values sampled from a standard normal distribution (mean=0, standard deviation=1) by default.
matrix_A.numpy() retrieves the values of the TensorFlow tensor as a NumPy array, allowing you to display the matrix.
# In[ ]:





# 2. Create a Gaussian matrix B with dimensions x, using TensorFlow's truncated_normal function. Display the values of matrix B.
To create a matrix with random values from a truncated Gaussian (normal) distribution using TensorFlow's tf.random.truncated_normal function, you can specify the shape and the mean and standard deviation of the distribution. Here's how you can create and display such a matrix B with a specified shape:
import tensorflow as tf
# Define the shape of the matrix B
shape_x = (3, 4)  # Replace with your desired shape
# Define mean and standard deviation for the truncated normal distribution
mean = 0.0
stddev = 1.0
# Create a truncated normal matrix B
matrix_B = tf.random.truncated_normal(shape=shape_x, mean=mean, stddev=stddev)
# Display the values of matrix B
print("Matrix B:")
print(matrix_B.numpy())
In this code:
shape_x defines the desired shape of the matrix B. You can replace (3, 4) with the dimensions you want.
mean and stddev define the mean and standard deviation of the truncated normal distribution. Adjust these values as needed for your specific use case.
tf.random.truncated_normal generates a matrix with values sampled from a truncated normal distribution with the specified mean and standard deviation.
matrix_B.numpy() retrieves the values of the TensorFlow tensor as a NumPy array for displaying the matrix.
Replace (3, 4) with the desired dimensions for your matrix B.
# In[ ]:





# 3. Create a matrix C with dimensions 2x2, where the values are drawn from a normal distribution with a mean of 2 and a standard deviation of 0.5, using TensorFlow's random.normal function. Display the values of matrix C.
To create a 2x2 matrix C with values drawn from a normal distribution with a mean of 2 and a standard deviation of 0.5 using TensorFlow's random.normal function, you can follow these steps in Python:
import tensorflow as tf

# Define the mean and standard deviation
mean = 2.0
stddev = 0.5

# Create a 2x2 matrix C with random values from the normal distribution
C = tf.random.normal([2, 2], mean=mean, stddev=stddev)

# Display the values of matrix C
print(C)
Make sure you have TensorFlow installed in your Python environment to run this code. This will create a 2x2 matrix C with the specified properties and display its values.
# In[ ]:





# 4. Perform matrix addition between matrix A and matrix B, and store the result in matrix D.
To perform matrix addition between two matrices A and B and store the result in matrix D, you can use Python with a library like NumPy. Here's how you can do it:
import numpy as np

# Define matrices A and B (2x2 matrices for example)
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Perform matrix addition
D = A + B

# Display matrix D
print("Matrix D (Result of Addition):\n", D)
In this example, we first define matrices A and B with their respective values. Then, we perform matrix addition by simply using the + operator, and the result is stored in matrix D. Finally, we display the result by printing matrix D.
# In[ ]:





# 5. Perform matrix multiplication between matrix C and matrix D, and store the result in matrix E.
To perform matrix multiplication between matrix C and matrix D and store the result in matrix E, you can use Python with a library like NumPy. Here's how you can do it:
import numpy as np
# Define matrix C (2x2) and matrix D (2x2)
C = np.array([[1.5, 2.5],
              [3.5, 4.5]])

D = np.array([[5, 6],
              [7, 8]])

# Perform matrix multiplication
E = np.dot(C, D)
# Display matrix E
print("Matrix E (Result of Multiplication):\n", E)
In this example, we first define matrices C and D with their respective values. Then, we perform matrix multiplication using np.dot or the @ operator in Python, and the result is stored in matrix E. Finally, we display the result by printing matrix E.
# In[ ]:





# # Task 2: Performing Additional Matrix Operations

# 1. Create a matrix F with dimensions 2x2, initialized with random values using TensorFlow's random_uniform function.
To create a 2x2 matrix F with random values using TensorFlow's random_uniform function, you can follow these steps in Python:
import tensorflow as tf
# Define the shape of the matrix
shape = (2, 2)
# Create a 2x2 matrix F with random values
F = tf.random.uniform(shape=shape)
# Display the values of matrix F
print(F)
This code snippet will create a 2x2 matrix F with random values distributed uniformly between 0 and 1 using TensorFlow's random_uniform function and then print the values of matrix F. Make sure you have TensorFlow installed in your Python environment to run this code.
# In[ ]:





# 2. Calculate the transpose of matrix F and store the result in matrix G.
To calculate the transpose of matrix F and store the result in matrix G using TensorFlow in Python, you can follow these steps:
import tensorflow as tf
# Define the original matrix F
F = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])

# Calculate the transpose and store it in matrix G
G = tf.transpose(F)
# Display the values of matrix G
print(G)
In this code, we first define matrix F with the values you want to transpose. Then, we use TensorFlow's tf.transpose function to calculate the transpose of F and store it in matrix G. Finally, we print the values of matrix G, which will be the transpose of matrix F.
# 3. Calculate the element-wise exponential of matrix F and store the result in matrix H.
To calculate the element-wise exponential of matrix F and store the result in matrix H using TensorFlow in Python, you can follow these steps:
import tensorflow as tf
# Define the original matrix F
F = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])
# Calculate the element-wise exponential and store it in matrix H
H = tf.math.exp(F)
# Display the values of matrix H
print(H)
In this code, we first define matrix F with the values you want to exponentiate element-wise. Then, we use TensorFlow's tf.math.exp function to calculate the element-wise exponential of F and store it in matrix H. Finally, we print the values of matrix H, which will be the element-wise exponential of matrix F.
# In[ ]:





# 4. Create a matrix I by concatenating matrix F and matrix G horizontally.
To concatenate matrix F and matrix G horizontally (side by side) and create a new matrix I using TensorFlow in Python, you can use the tf.concat function. Here's how you can do it:
import tensorflow as tf
# Define the original matrices F and G
F = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])
G = tf.constant([[5.0, 6.0],
                 [7.0, 8.0]])
# Concatenate F and G horizontally to create matrix I
I = tf.concat([F, G], axis=1)
# Display the values of matrix I
print(I)
In this code, we first define matrices F and G with their respective values. Then, we use TensorFlow's tf.concat function to concatenate them horizontally along axis=1 (columns), creating matrix I. Finally, we print the values of matrix I, which will be the result of the horizontal concatenation of F and G.
# In[ ]:





# 5. Create a matrix J by concatenating matrix F and matrix H vertically.
To concatenate matrix F and matrix H vertically (on top of each other) and create a new matrix J using TensorFlow in Python, you can use the tf.concat function with axis=0. Here's how you can do it:
import tensorflow as tf
# Define the original matrices F and H
F = tf.constant([[1.0, 2.0],
                 [3.0, 4.0]])
H = tf.math.exp(F)  # Calculate the element-wise exponential of F
# Concatenate F and H vertically to create matrix J
J = tf.concat([F, H], axis=0)
# Display the values of matrix J
print(J)
In this code, we first define matrix F with the original values, and then we calculate matrix H, which is the element-wise exponential of matrix F using tf.math.exp. After that, we use TensorFlow's tf.concat function to concatenate matrices F and H vertically along axis=0 (rows), creating matrix J. Finally, we print the values of matrix J, which will be the result of the vertical concatenation of F and H.
# In[ ]:





# #  <P style="color:green"> THAT'S ALL, THANK YOU    </p>
