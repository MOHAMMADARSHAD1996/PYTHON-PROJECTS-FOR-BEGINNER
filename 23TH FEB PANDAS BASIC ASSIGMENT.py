#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Q1. Create a Pandas Series that contains the following data: 4, 8, 15, 16, 23, and 42. Then, print the series.

# In[1]:


import pandas as pd

data = [4, 8, 15, 16, 23, 42]
series = pd.Series(data)

print(series)
#When you run this code, it will create a Pandas Series with the provided data and then print it to the console.


# Q2. Create a variable of list type containing 10 elements in it, and apply pandas.Series function on the
# variable print it.

# In[4]:


# Certainly! Here's an example of how you can create a list variable with 10 elements and 
# then convert it into a Pandas Series using the pd.Series function:
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
series = pd.Series(my_list)

print(series)
"""In this code, a list named my_list is created containing 10 elements,
   and then the pd.Series function is used to convert the list into a Pandas Series.
   Finally, the Series is printed to the console."""


# Q3. Create a Pandas DataFrame that contains the following data:
# Name
# Alice
# Bob
# Claire
# 
# Age
# 25
# 30
# 27
# 
# Gender
# Female
# Male
# Female
# 
# Then, print the DataFrame.

# In[5]:


data = {
    'Name': ['Alice', 'Bob', 'Claire'],
    'Age': [25, 30, 27],
    'Gender': ['Female', 'Male', 'Female']
 }

df = pd.DataFrame(data)

print(df)
"""This code creates a dictionary data with the columns 'Name', 'Age', and 'Gender', and 
their corresponding values. Then, the pd.DataFrame function is used to convert this dictionary into a Pandas DataFrame. 
Finally, the DataFrame is printed to the console."""


# Q4. What is ‘DataFrame’ in pandas and how is it different from pandas.series? Explain with an example.
A DataFrame in Pandas is a 2-dimensional labeled data structure that can hold heterogeneous data. It's essentially a table with rows and columns, where each column is a Pandas Series. A Series, on the other hand, is a 1-dimensional labeled array that can hold data of any type.

Let's use an example to illustrate the difference:

Suppose we have data about students:

Student ID	Name	Age	Grade
1	Alice	25	A
2	Bob	30	B
3	Claire	27	A
Here's how we can represent this data using DataFrames and Series:
Using DataFrame
data = {
    'Student ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Claire'],
    'Age': [25, 30, 27],
    'Grade': ['A', 'B', 'A']
}
df = pd.DataFrame(data)

Using Series
student_ids = pd.Series([1, 2, 3], name='Student ID')
names = pd.Series(['Alice', 'Bob', 'Claire'], name='Name')
ages = pd.Series([25, 30, 27], name='Age')
grades = pd.Series(['A', 'B', 'A'], name='Grade')
The DataFrame df and the Series student_ids, names, ages, and grades contain the same data. However, the DataFrame allows you to organize and manage the data in a tabular format, while the Series are individual columns that can be used separately but are often part of a DataFrame.
# Q5. What are some common functions you can use to manipulate data in a Pandas DataFrame? Can
# you give an example of when you might use one of these functions?
Certainly! Pandas offers a wide range of functions to manipulate data in a DataFrame. Here are some common functions and methods, along with examples of when you might use them:
head() and tail(): These functions allow you to view the first few or last few rows of a DataFrame.
df.head()  # View the first 5 rows
df.tail(10)  # View the last 10 rows
info(): This method provides information about the DataFrame, including data types and non-null counts.

df.info()
describe(): This method provides summary statistics for numerical columns in the DataFrame.


df.describe()
shape: This attribute returns the dimensions (rows, columns) of the DataFrame.

rows, cols = df.shape
loc[] and iloc[]: These indexers allow you to access rows and columns using labels or integer indices, respectively.

df.loc[2, 'Name']  # Access data at row 2, column 'Name'
df.iloc[0:3, 1:3]  # Access rows 0 to 2, columns 1 to 2 using integer indexing
groupby(): This method is used for grouping and aggregating data based on specific columns.

grouped = df.groupby('Grade')
avg_age_by_grade = grouped['Age'].mean()
sort_values(): This method sorts the DataFrame based on one or more columns.

df_sorted = df.sort_values(by='Age', ascending=False)
drop(): This method is used to drop rows or columns from the DataFrame.

df_dropped = df.drop(columns=['Grade'])  # Drop the 'Grade' column
fillna(): This method fills missing values with specified values or strategies.

df_filled = df.fillna(value=0)  # Fill NaN values with 0
apply(): This method applies a function along an axis (rows or columns) of the DataFrame.

def double_age(age):
    return age * 2

df['Double_Age'] = df['Age'].apply(double_age)
These are just a few examples of the many functions and methods available in Pandas to manipulate data within a DataFrame. Depending on your specific analysis or data transformation needs, you can utilize these functions to clean, reshape, analyze, and visualize your data effectively.
# In[ ]:





# Q6. Which of the following is mutable in nature Series, DataFrame, Panel?
Among the options provided, both Series and DataFrame are mutable in nature in Pandas, while Panel is not.

Series: A Pandas Series is mutable, meaning you can modify its values after it's created. You can change values, add or remove elements, and perform various operations on the Series.

DataFrame: Similarly, a Pandas DataFrame is also mutable. You can change column values, add or drop columns, modify rows, and perform various data manipulation tasks.

Panel: However, the concept of a Panel has been deprecated in recent versions of Pandas (since version 0.25.0). It used to be a 3D data structure representing data using multiple DataFrames, but it's no longer considered a core data structure in Pandas. Instead, you are encouraged to work with hierarchical indexing on DataFrames for similar purposes.

So, among the options, only Panel is not mutable, and both Series and DataFrame are mutable.
# In[ ]:





# Q7. Create a DataFrame using multiple Series. Explain with an example.
Certainly! You can create a DataFrame using multiple Series by passing a dictionary of Series to the pd.DataFrame() constructor. Each Series becomes a column in the DataFrame. Here's an example:

Suppose you have data about students:

Student ID	Name	Age	Grade
1	Alice	25	A
2	Bob	30	B
3	Claire	27	A
You can create a DataFrame using multiple Series like this:
# In[6]:


# Create individual Series
student_ids = pd.Series([1, 2, 3], name='Student ID')
names = pd.Series(['Alice', 'Bob', 'Claire'], name='Name')
ages = pd.Series([25, 30, 27], name='Age')
grades = pd.Series(['A', 'B', 'A'], name='Grade')

# Create a DataFrame using the Series
data = {
    'Student ID': student_ids,
    'Name': names,
    'Age': ages,
    'Grade': grades
}

df = pd.DataFrame(data)

print(df)

In this example, each Series (student_ids, names, ages, grades) represents a column in the DataFrame. The dictionary data maps column names to their corresponding Series, and the pd.DataFrame() constructor creates the DataFrame using this data. The resulting DataFrame contains the student information with columns "Student ID", "Name", "Age", and "Grade".
# In[ ]:





# #  <P style="color:GREEN"> Thank You ,That's All </p>
