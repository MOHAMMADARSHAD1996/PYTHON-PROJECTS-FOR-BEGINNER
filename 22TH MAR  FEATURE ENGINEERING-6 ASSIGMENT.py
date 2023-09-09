#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple">  FEATURE ENGINEERING-6</p>

# Q1. Pearson correlation coefficient is a measure of the linear relationship between two variables. Suppose 
# you have collected data on the amount of time students spend studying for an exam and their final exam 
# scores. Calculate the Pearson correlation coefficient between these two variables and interpret the result.

# In[1]:


# To calculate the Pearson correlation coefficient (also known as Pearson's r) between two variables, you can use the following formula:

# �
# =
# ∑
# (
# �
# �
# −
# �
# ˉ
# )
# (
# �
# �
# −
# �
# ˉ
# )
# ∑
# (
# �
# �
# −
# �
# ˉ
# )
# 2
# ⋅
# ∑
# (
# �
# �
# −
# �
# ˉ
# )
# 2
# r= 
# ∑(X 
# i
# ​
#  − 
# X
# ˉ
#  ) 
# 2
#  ⋅∑(Y 
# i
# ​
#  − 
# Y
# ˉ
#  ) 
# 2
 
# ​
 
# ∑(X 
# i
# ​
#  − 
# X
# ˉ
#  )(Y 
# i
# ​
#  − 
# Y
# ˉ
#  )
# ​
 

# Where:

# �
# r is the Pearson correlation coefficient.
# �
# �
# X 
# i
# ​
#   and 
# �
# �
# Y 
# i
# ​
#   are individual data points for the two variables.
# �
# ˉ
# X
# ˉ
#   and 
# �
# ˉ
# Y
# ˉ
#   are the means (averages) of the two variables.
# The sums in the denominator are taken over all data points.
# Let's assume you have collected data on the time students spend studying (in hours) and their corresponding final exam scores (out of 100). Here's how you can calculate and interpret the Pearson correlation coefficient:

import numpy as np

# Sample data (replace with your actual data)
study_time = [10, 15, 20, 25, 30]
exam_scores = [70, 75, 80, 85, 90]

# Calculate the means
mean_study_time = np.mean(study_time)
mean_exam_scores = np.mean(exam_scores)

# Calculate the Pearson correlation coefficient
numerator = sum((x - mean_study_time) * (y - mean_exam_scores) for x, y in zip(study_time, exam_scores))
denominator_x = sum((x - mean_study_time) ** 2 for x in study_time)
denominator_y = sum((y - mean_exam_scores) ** 2 for y in exam_scores)
pearson_r = numerator / (np.sqrt(denominator_x * denominator_y))

# Interpret the result
print(f"Pearson Correlation Coefficient (r): {pearson_r:.2f}")

if pearson_r > 0:
    print("There is a positive linear relationship between study time and exam scores.")
elif pearson_r < 0:
    print("There is a negative linear relationship between study time and exam scores.")
else:
    print("There is no linear relationship between study time and exam scores.")
# In this example, we first calculate the Pearson correlation coefficient, and then we interpret the result:

# If 
# �
# r is close to 1, it indicates a strong positive linear relationship, meaning that as students spend more time studying, their exam scores tend to be higher.

# If 
# �
# r is close to -1, it indicates a strong negative linear relationship, meaning that as students spend more time studying, their exam scores tend to be lower.

# If 
# �
# r is close to 0, it suggests little to no linear relationship between the two variables.

# The interpretation will depend on the actual value of 
# �
# r that you calculate from your data.


# In[ ]:





# Q2. Spearman's rank correlation is a measure of the monotonic relationship between two variables. 
# Suppose you have collected data on the amount of sleep individuals get each night and their overall job 
# satisfaction level on a scale of 1 to 10. Calculate the Spearman's rank correlation between these two 
# variables and interpret the result.

# In[2]:


# Spearman's rank correlation coefficient (Spearman's rho or ρ) is a non-parametric measure of the strength and direction of the monotonic relationship between two variables. It assesses whether one variable tends to increase or decrease as the other does, without assuming a linear relationship. Here's how to calculate and interpret Spearman's rank correlation for the provided data on sleep and job satisfaction:

# Rank the Data: First, you need to rank both variables separately. Assign a rank to each data point for each variable, with 1 being the lowest rank and N being the highest rank (N is the number of data points). Ties should be given the average of the ranks.

# Calculate the Differences: For each pair of data points, calculate the difference in ranks for both variables.

# Square the Differences: Square each of the differences calculated in step 2.

# Sum the Squares: Sum up all the squared differences.

# Use the Formula: Use the formula for Spearman's rank correlation:

# �
# =
# 1
# −
# 6
# ∑
# �
# �
# 2
# �
# (
# �
# 2
# −
# 1
# )
# ρ=1− 
# n(n 
# 2
#  −1)
# 6∑d 
# i
# 2
# ​
 
# ​
 

# Where:

# �
# ρ is the Spearman's rank correlation coefficient.
# �
# �
# d 
# i
# ​
#   are the differences in ranks for each pair of data points.
# �
# n is the number of data points.
# Now, let's calculate Spearman's rank correlation and interpret the result using Python:

from scipy.stats import spearmanr

# Sample data (replace with your actual data)
sleep_hours = [7, 6, 8, 5, 7]
job_satisfaction = [8, 6, 9, 5, 7]

# Calculate Spearman's rank correlation
rho, _ = spearmanr(sleep_hours, job_satisfaction)

# Interpret the result
print(f"Spearman's Rank Correlation (rho): {rho:.2f}")

if rho > 0:
    print("There is a positive monotonic relationship between sleep hours and job satisfaction.")
elif rho < 0:
    print("There is a negative monotonic relationship between sleep hours and job satisfaction.")
else:
    print("There is no monotonic relationship between sleep hours and job satisfaction.")
# In this code, we use the spearmanr function from the SciPy library to calculate Spearman's rank correlation coefficient.

# If 
# �
# ρ is close to 1, it indicates a strong positive monotonic relationship, suggesting that as individuals get more sleep, their job satisfaction tends to increase.

# If 
# �
# ρ is close to -1, it indicates a strong negative monotonic relationship, suggesting that as individuals get more sleep, their job satisfaction tends to decrease.

# If 
# �
# ρ is close to 0, it suggests little to no monotonic relationship between the two variables.

# Interpreting the result will depend on the actual value of 
# �
# ρ that you calculate from your data.


# In[ ]:





# Q3. Suppose you are conducting a study to examine the relationship between the number of hours of 
# exercise per week and body mass index (BMI) in a sample of adults. You collected data on both variables 
# for 50 participants. Calculate the Pearson correlation coefficient and the Spearman's rank correlation 
# between these two variables and compare the results.
# To examine the relationship between the number of hours of exercise per week and body mass index (BMI) in a sample of adults, you can calculate both the Pearson correlation coefficient (for linear relationships) and the Spearman's rank correlation (for monotonic relationships) to understand different aspects of the relationship. Here's how you can calculate and compare these two correlation coefficients using Python:
# Assuming you have collected data for both variables, let's calculate the Pearson and Spearman correlations and compare the results:

from scipy.stats import pearsonr, spearmanr

# Sample data (replace with your actual data)
exercise_hours = [2, 3, 4, 1, 2, 5, 3, 4, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6,
                  2, 3, 4, 1, 2, 5, 3, 4, 6, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6,
                  2, 3, 4, 1, 2, 5, 3, 4, 6, 1]

bmi = [25.3, 26.5, 24.8, 28.1, 25.6, 23.9, 26.2, 24.7, 23.1, 29.0, 27.8, 26.3,
       24.5, 23.7, 28.7, 26.8, 25.9, 24.1, 23.5, 22.4, 25.2, 26.4, 24.9, 28.2,
       27.3, 23.6, 25.8, 26.1, 22.7, 24.0, 27.1, 25.0, 23.4, 28.8, 26.7, 24.6,
       25.7, 22.9, 23.3, 27.9, 26.9, 25.4, 24.2, 28.4, 27.6,24.5, 23.2, 25.1]

# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(exercise_hours, bmi)

# Calculate Spearman's rank correlation
spearman_corr, _ = spearmanr(exercise_hours, bmi)

# Compare the results
print(f"Pearson Correlation Coefficient: {pearson_corr:.2f}")
print(f"Spearman's Rank Correlation: {spearman_corr:.2f}")

# Interpret the results
if abs(pearson_corr) > abs(spearman_corr):
    print("The Pearson correlation coefficient suggests a stronger linear relationship.")
elif abs(pearson_corr) < abs(spearman_corr):
    print("Spearman's rank correlation suggests a stronger monotonic relationship.")
else:
    print("Both correlation coefficients provide similar results.")
# In this code, we calculate both the Pearson and Spearman correlations between exercise hours and BMI. We then compare the results and interpret them.
# If the Pearson correlation coefficient is significantly higher than the Spearman correlation coefficient, it suggests that there is a strong linear relationship between the variables.
# If the Spearman's rank correlation is significantly higher than the Pearson coefficient, it indicates a strong monotonic relationship between the variables, which may not necessarily be linear.
# If both correlation coefficients are similar, it suggests consistency between the linear and monotonic relationships.
# The interpretation will depend on the actual values of the correlation coefficients calculated from your data.
# In[ ]:





# Q4. A researcher is interested in examining the relationship between the number of hours individuals 
# spend watching television per day and their level of physical activity. The researcher collected data on 
# both variables from a sample of 50 participants. Calculate the Pearson correlation coefficient between 
# these two variables.

# In[7]:


# To calculate the Pearson correlation coefficient between the number of hours individuals spend watching television per day and their level of physical activity, you can use the Pearson correlation formula. Here's how you can do it in Python:
# Assuming you have collected data for both variables, let's calculate the Pearson correlation coefficient:
# Sample data (replace with your actual data)
tv_hours = [3, 2, 4, 5, 2, 1, 6, 3, 4, 2, 1, 5, 3, 2, 4, 6, 1, 2, 3, 4,
            3, 5, 2, 1, 4, 2, 3, 6, 5, 1, 2, 4, 5, 2, 3, 4, 6, 1, 2, 3,
            3, 4, 2, 1, 5, 2, 6, 4, 3, 2]

physical_activity_level = [2, 3, 1, 4, 3, 5, 1, 2, 4, 3, 5, 2, 3, 4, 1, 1, 5,
                           3, 2, 4, 3, 2, 5, 1, 4, 2, 3, 1, 5, 4, 3, 2, 2, 1,
                           4, 5, 3, 2, 1, 4, 3, 5, 2, 3, 4, 1, 1, 5, 3, 2]

# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(tv_hours, physical_activity_level)

# Interpret the result
print(f"Pearson Correlation Coefficient: {pearson_corr:.2f}")

if pearson_corr > 0:
    print("There is a positive correlation between TV hours and physical activity.")
elif pearson_corr < 0:
    print("There is a negative correlation between TV hours and physical activity.")
else:
    print("There is little to no correlation between TV hours and physical activity.")
# In this code, we calculate the Pearson correlation coefficient between the number of hours individuals spend watching television per day and their level of physical activity.
# If the Pearson correlation coefficient (
# �
# r) is positive and close to 1, it suggests a positive correlation, meaning that as the number of TV hours increases, physical activity tends to increase as well.
# If 
# �
# r is negative and close to -1, it suggests a negative correlation, indicating that as TV hours increase, physical activity tends to decrease.
# If 
# �
# r is close to 0, it suggests little to no correlation, meaning there is no strong linear relationship between TV hours and physical activity.
# The interpretation will depend on the actual value of 
# �
# r that you calculate from your data.


# In[ ]:





# Q5. A survey was conducted to examine the relationship between age and preference for a particular 
# brand of soft drink. The survey results are shown below:
# Age(Years)
# 25 Coke
# 42 Pepsi
# 37
# 19
# 31
# 28
# Mountain dew
# Coke
# Pepsi
# Coke
# It appears that you have survey data related to age and preference for a particular brand of soft drink. However, the data you've provided is not in a format that allows for direct analysis. It seems like you have a list of ages followed by brand names (Coke, Pepsi, Mountain Dew) but without clear labels or structure.

# To analyze the relationship between age and brand preference, you should have a structured dataset that associates each age with a specific brand choice. It should look something like this:

# python
Age (Years)   Brand Preference
25            Coke
42            Pepsi
37            Pepsi
19            Mountain Dew
31            Coke
28            Pepsi
...
# Once you have a structured dataset like the one above, you can calculate summary statistics and perform statistical analyses, such as correlation or chi-square tests, to examine the relationship between age and brand preference.
# If you have the raw data in a different format or if you have any specific questions about the analysis, please provide additional details, and I'd be happy to assist you further
# In[ ]:




6. A company is interested in examining the relationship between the number of sales calls made per day 
and the number of sales made per week. The company collected data on both variables from a sample of 
30 sales representatives. Calculate the Pearson correlation coefficient between these two variables.
# In[9]:


# To calculate the Pearson correlation coefficient between the number of sales calls made per day and the number of sales made per week for a sample of 30 sales representatives, you can use Python's NumPy and SciPy libraries. Here's how you can do it:

# Assuming you have collected data for both variables, let's calculate the Pearson correlation coefficient:
import numpy as np
from scipy.stats import pearsonr

# Sample data (replace with your actual data)
sales_calls_per_day = [10, 12, 15, 8, 9, 14, 11, 13, 16, 9, 10, 12, 14, 17, 8, 9, 11, 13, 15, 18,
                        10, 12, 15, 8, 9, 14, 11, 13, 16, 9]

sales_per_week = [50, 55, 60, 45, 48, 62, 52, 58, 64, 46, 50, 54, 60, 68, 44, 47, 51, 56, 59, 70,
                  51, 53, 61, 43, 49, 63, 53, 57, 66, 45]

# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(sales_calls_per_day, sales_per_week)

# Interpret the result
print(f"Pearson Correlation Coefficient: {pearson_corr:.2f}")

if pearson_corr > 0:
    print("There is a positive correlation between sales calls per day and sales per week.")
elif pearson_corr < 0:
    print("There is a negative correlation between sales calls per day and sales per week.")
else:
    print("There is little to no correlation between sales calls per day and sales per week.")
# In this code, we calculate the Pearson correlation coefficient (
# �
# r) between the number of sales calls made per day and the number of sales made per week.

# If 
# �
# r is positive and close to 1, it suggests a positive correlation, indicating that as the number of sales calls per day increases, the number of sales per week tends to increase as well.

# If 
# �
# r is negative and close to -1, it suggests a negative correlation, indicating that as the number of sales calls per day increases, the number of sales per week tends to decrease.

# If 
# �
# r is close to 0, it suggests little to no correlation, meaning there is no strong linear relationship between sales calls per day and sales per week.

# The interpretation will depend on the actual value of 
# �
# r that you calculate from your data.






# #  <P style="color:green">  THANK YOU , THAT'S ALL   </p>
