#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Q1: What is Matplotlib? Why is it used? Name five plots that can be plotted using the Pyplot module of
# Matplotlib.
Matplotlib is a popular Python library for creating static, animated, and interactive visualizations in various formats such as line plots, scatter plots, bar charts, histograms, and more. It provides a wide range of tools for constructing and customizing data visualizations. Matplotlib is often used for data analysis, data visualization, and scientific plotting.

Five common types of plots that can be created using the Pyplot module of Matplotlib include:

Line Plot: Line plots are used to visualize data points connected by lines, making them suitable for showing trends or continuous data.

Scatter Plot: Scatter plots display individual data points as markers on a two-dimensional plane, useful for exploring relationships between two variables.

Bar Chart: Bar charts or bar plots represent data using rectangular bars of varying heights or lengths, commonly used for comparing categories or showing discrete data.

Histogram: Histograms are used to visualize the distribution of a single variable by dividing the data into bins and counting the number of data points in each bin.

Pie Chart: Pie charts represent data in a circular format, with each category represented as a slice of the pie, showing the proportion of each category relative to the whole.

These are just a few examples, and Matplotlib supports many other types of plots and customizations to suit various data visualization needs.
# In[ ]:





# Q2: What is a scatter plot? Use the following code to generate data for x and y. Using this generated data
# plot a scatter plot.
# 
# import numpy as np
# 
# np.random.seed(3)
# 
# x = 3 + np.random.normal(0, 2, 50)
# 
# y = 3 + np.random.normal(0, 2, len(x))
# 
# Note: Also add title, xlabel, and ylabel to the plot.

# In[2]:


""""A scatter plot is a type of data visualization 
that displays individual data points as markers (usually dots) on a two-dimensional plane.
It is particularly useful for visualizing the relationship between two variables, showing patterns,
clusters, or trends in the data.
To create a scatter plot using the provided data and add a title, xlabel, and ylabel to it,
you can use the Matplotlib library. Here's the code to generate the scatter plot:"""
import matplotlib.pyplot as plt
import numpy as np
# Set a random seed for reproducibility
np.random.seed(3)
# Generate data for x and y
x = 3 + np.random.normal(0, 2, 50)
y = 3 + np.random.normal(0, 2, len(x))
# Create the scatter plot
plt.scatter(x, y)
# Add title, xlabel, and ylabel
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# Show the plot
plt.show()
# In this code:
# We import the necessary libraries, including NumPy and Matplotlib.
# We generate random data for x and y using NumPy's np.random.normal function.
# We create a scatter plot using plt.scatter(x, y) where x and y are the data arrays.
# We add a title using plt.title, xlabel using plt.xlabel, and ylabel using plt.ylabel.
# Finally, we display the plot using plt.show().
# Running this code will generate a scatter plot with the provided data and labels.


# In[ ]:




Q3: Why is the subplot() function used? Draw four line plots using the subplot() function.
Use the following data:
import numpy as np
For line 1: x = np.array([0, 1, 2, 3, 4, 5]) and y = np.array([0, 100, 200, 300, 400, 500])
For line 2: x = np.array([0, 1, 2, 3, 4, 5]) and y = np.array([50, 20, 40, 20, 60, 70])
For line 3: x = np.array([0, 1, 2, 3, 4, 5]) and y = np.array([10, 20, 30, 40, 50, 60])
For line 4: x = np.array([0, 1, 2, 3, 4, 5]) and y = np.array([200, 350, 250, 550, 450, 150])
# In[3]:


"""The subplot() function in Matplotlib is used to create multiple subplots (smaller plots) within a single figure.
It allows you to arrange multiple plots in a grid-like structure,
making it easier to visualize and compare different datasets or aspects of your data.
Here's how you can use the subplot() function to create four line plots using the provided data:"""
# Data for line 1
x1 = np.array([0, 1, 2, 3, 4, 5])
y1 = np.array([0, 100, 200, 300, 400, 500])
# Data for line 2
x2 = np.array([0, 1, 2, 3, 4, 5])
y2 = np.array([50, 20, 40, 20, 60, 70])
# Data for line 3
x3 = np.array([0, 1, 2, 3, 4, 5])
y3 = np.array([10, 20, 30, 40, 50, 60])
# Data for line 4
x4 = np.array([0, 1, 2, 3, 4, 5])
y4 = np.array([200, 350, 250, 550, 450, 150])
# Create a figure with 2x2 grid of subplots
plt.figure(figsize=(10, 8))
# Plot line 1
plt.subplot(2, 2, 1)
plt.plot(x1, y1)
plt.title('Line 1')
# Plot line 2
plt.subplot(2, 2, 2)
plt.plot(x2, y2)
plt.title('Line 2')
# Plot line 3
plt.subplot(2, 2, 3)
plt.plot(x3, y3)
plt.title('Line 3')
# Plot line 4
plt.subplot(2, 2, 4)
plt.plot(x4, y4)
plt.title('Line 4')
# Adjust layout
plt.tight_layout()
# Show the plots
plt.show()
# In this code:
# We define the data for four different lines.
# We create a figure with a 2x2 grid of subplots using plt.subplot().
# We plot each line on its corresponding subplot with titles.
# plt.tight_layout() is used to ensure proper spacing between subplots.
# Finally, we use plt.show() to display the plots.


# In[ ]:




Q4: What is a bar plot? Why is it used? Using the following data plot a bar plot and a horizontal bar plot.
import numpy as np
company = np.array(["Apple", "Microsoft", "Google", "AMD"])
profit = np.array([3000, 8000, 1000, 10000])
# In[4]:


# A bar plot (also known as a bar chart or bar graph) is a type of data visualization that represents
# categorical data with rectangular bars. 
# Each bar's length or height is proportional to the value it represents,
# making it a suitable choice for displaying comparisons between different categories or groups.
# A bar plot is used to:
# Compare Data: It helps in comparing data across different categories or groups visually.
# Show Distribution: It can display the distribution of a single categorical variable or the relationship between two categorical variables.
# Highlight Differences: It is effective in highlighting differences or trends in data.
# To create a bar plot and a horizontal bar plot using the provided data, you can use Matplotlib. Here's the code to do that:
company = np.array(["Apple", "Microsoft", "Google", "AMD"])
profit = np.array([3000, 8000, 1000, 10000])
# Create a bar plot
plt.figure(figsize=(8, 5))
plt.bar(company, profit)
plt.title('Bar Plot')
plt.xlabel('Company')
plt.ylabel('Profit (in millions)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
# Create a horizontal bar plot
plt.figure(figsize=(8, 5))
plt.barh(company, profit)
plt.title('Horizontal Bar Plot')
plt.xlabel('Profit (in millions)')
plt.ylabel('Company')
plt.tight_layout()
# Show the plots
plt.show()
# In this code:
# We use plt.bar() to create a vertical bar plot and plt.barh() to create a horizontal bar plot.
# The company array is used for the x-axis (categories), and the profit array is used for the heights of the bars.
# We add titles, labels, and adjust the layout for both plots.
# plt.xticks(rotation=45) is used to rotate the x-axis labels in the vertical bar plot for better readability.
# Finally, we use plt.show() to display both the vertical and horizontal bar plots.


# In[ ]:




Q5: What is a box plot? Why is it used? Using the following data plot a box plot.
box1 = np.random.normal(100, 10, 200)
box2 = np.random.normal(90, 20, 200)
# In[5]:


# A box plot (also known as a box-and-whisker plot) is a statistical visualization tool that provides a graphical summary of the distribution of a dataset. It is particularly useful for displaying the spread, central tendency, and potential outliers within a dataset. A box plot consists of several components:
# Box: The box represents the interquartile range (IQR), which contains the middle 50% of the data. The top and bottom edges of the box indicate the first quartile (Q1) and third quartile (Q3), respectively, while the line inside the box represents the median (Q2).
# Whiskers: The whiskers extend from the edges of the box to the minimum and maximum values within a certain range. They can also indicate the potential presence of outliers.
# Outliers: Individual data points that fall outside the whiskers are considered outliers and are often displayed individually.
# A box plot is used to:
# Summarize Data Distribution: It provides a concise summary of the distribution of data, including central tendency and spread.
# Identify Outliers: It can help identify potential outliers in a dataset.
# Compare Distributions: Box plots can be used to compare the distributions of multiple datasets.
# Here's how you can create a box plot using the provided data:
# Generate random data for box1 and box2
box1 = np.random.normal(100, 10, 200)
box2 = np.random.normal(90, 20, 200)

# Create a box plot
plt.figure(figsize=(8, 5))
plt.boxplot([box1, box2], labels=['Box 1', 'Box 2'])
plt.title('Box Plot')
plt.xlabel('Boxes')
plt.ylabel('Values')
plt.grid(True)
# Show the plot
plt.show()
# In this code:
# We use plt.boxplot() to create a box plot with two datasets, box1 and box2.
# We label the boxes as "Box 1" and "Box 2" using the labels parameter.
# We add titles, labels, and enable gridlines for the plot.
# Finally, we use plt.show() to display the box plot with the two datasets.


# In[ ]:





# #  <P style="color:GREEN"> Thank You ,That's All </p>
