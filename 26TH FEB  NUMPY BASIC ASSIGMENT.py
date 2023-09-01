#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Consider the below code to answer further questions:
# 
# 
# import numpy as np
# 
# list_ = [ ‘1’ , ’2’ , ‘3’ , ‘4’ , ‘5’ ]
# 
# array_list = np.array(object = list_)
# Q1. Is there any difference in the data type of variables list_ and array_list? If there is then write a code 
# to print the data types of both the variables.

# In[1]:


"""Yes, there is a difference in the data type of the list_ and array_list variables.
list_ is a Python list, and array_list is a NumPy array. 
You can use the type() function to check the data types of these variables.
Here's the code to do that:"""
import numpy as np

list_ = ['1', '2', '3', '4', '5']
array_list = np.array(object=list_)

# Check the data types
print("Data type of 'list_':", type(list_))
print("Data type of 'array_list':", type(array_list))
"""When you run this code, it will print the data types of both list_ and array_list,
which will demonstrate the difference in data types between the two variables."""


# In[ ]:





# Q2. Write a code to print the data type of each and every element of both the variables list_ and 
# arra_list.

# In[2]:


"""To print the data type of each element in both the list_ and array_list variables, 
you can use a loop to iterate through each element and 
apply the type() function to each element. Here's the code to do that:"""
list_ = ['1', '2', '3', '4', '5']
array_list = np.array(object=list_)

# Print data types of elements in 'list_'
print("Data types of elements in 'list_':")
for element in list_:
    print(type(element))

# Print data types of elements in 'array_list'
print("\nData types of elements in 'array_list':")
for element in array_list:
    print(type(element))
"""This code will iterate through each element in both list_ and array_list, and
it will print the data type of each individual element in both variables."""


# In[ ]:





# Q3. Considering the following changes in the variable, array_list:
# 
# array_list = np.array(object = list_, dtype = int)
# 
# Will there be any difference in the data type of the elements present in both the variables, list_ and 
# arra_list? If so then print the data types of each and every element present in both the variables, list_ 
# and arra_list.
# 
# Consider the below code to answer further questions:
# 
# import numpy as np
# 
# num_list = [ [ 1 , 2 , 3 ] , [ 4 , 5 , 6 ] ]
# 
# num_array = np.array(object = num_list)

# In[3]:


"""Yes, there will be a difference in the data types of the elements present in the list_ and array_list
variables after the change you mentioned. 
When you create array_list with the dtype=int argument, 
it explicitly specifies that the elements of the NumPy array should be of integer data type.
Here's the code to print the data types of each element present in both list_ and array_list:"""
list_ = ['1', '2', '3', '4', '5']
array_list = np.array(object=list_, dtype=int)

# Print data types of elements in 'list_'
print("Data types of elements in 'list_':")
for element in list_:
    print(type(element))

# Print data types of elements in 'array_list'
print("\nData types of elements in 'array_list':")
for element in array_list:
    print(type(element))
"""With this code, you will see that the elements in the list_ variable are of type string,
while the elements in the array_list variable are of type integer due to the specified dtype=int."""


# In[ ]:





# Q4. Write a code to find the following characteristics of variable, num_array:
# (i)	 shape
# (ii) size

# In[4]:


"""You can find the shape and size of the NumPy array num_array using the shape and size attributes, respectively.
Here's the code to do that:"""
num_list = [[1, 2, 3], [4, 5, 6]]
num_array = np.array(object=num_list)

# Find the shape of the array
array_shape = num_array.shape

# Find the size of the array
array_size = num_array.size

print("Shape of num_array:", array_shape)
print("Size of num_array:", array_size)
"""When you run this code, it will print the shape and size of the num_array variable, providing you with the characteristics of the array."""


# In[ ]:





# Q5. Write a code to create numpy array of 3*3 matrix containing zeros only, using a numpy array 
# creation function.
# [Hint: The size of the array will be 9 and the shape will be (3,3).]

# In[5]:


"""You can create a NumPy array of a 3x3 matrix containing zeros using the np.zeros function. 
Here's the code to do that:"""
# Create a 3x3 array of zeros
zero_matrix = np.zeros((3, 3))

print(zero_matrix)
"""This code uses np.zeros to create a 3x3 matrix filled with zeros and assigns it to the variable zero_matrix. 
The shape of the resulting array will be (3, 3), and all elements will be zeros."""


# In[ ]:





# Q6. Create an identity matrix of shape (5,5) using numpy functions?
# [Hint: An identity matrix is a matrix containing 1 diagonally and other elements will be 0.]

# In[ ]:





# #  <P style="color:GREEN"> Thank You ,That's All </p>
