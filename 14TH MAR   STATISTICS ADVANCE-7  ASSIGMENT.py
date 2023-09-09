#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> STATISTICS ADVANCE-7  </p>

# Q1. Write a Python function that takes in two arrays of data and calculates the F-value for a variance ratio 
# test. The function should return the F-value and the corresponding p-value for the test.

# In[1]:


#To calculate the F-value for a variance ratio test in Python, you can use the scipy.stats library,
#which provides a f_oneway function for one-way ANOVA (analysis of variance). This function can be used to perform the variance ratio test.
#Here's a Python function that takes two arrays of data and calculates the F-value and the corresponding p-value for the test:
import numpy as np
from scipy.stats import f_oneway

def calculate_f_value(data1, data2):
    # Ensure the input data are numpy arrays
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Perform the variance ratio test (F-test)
    f_statistic, p_value = f_oneway(data1, data2)
    
    return f_statistic, p_value

# Example usage:
data1 = [12, 14, 16, 18, 20]
data2 = [8, 10, 12, 14, 16]

f_stat, p_val = calculate_f_value(data1, data2)
print("F-Value:", f_stat)
print("P-Value:", p_val)
# In this example, data1 and data2 are two arrays containing the data you want to compare.
# The calculate_f_value function converts these data arrays into NumPy arrays and
# then uses f_oneway to perform the variance ratio test (F-test). It returns the F-value and the corresponding p-value.
# You can replace data1 and data2 with your own datasets to perform the test on your specific data.
# Make sure you have the scipy library installed. You can install it using pip if you haven't already:


# In[ ]:





# Q2. Given a significance level of 0.05 and the degrees of freedom for the numerator and denominator of an 
# F-distribution, write a Python function that returns the critical F-value for a two-tailed test.

# In[2]:


# To calculate the critical F-value for a two-tailed test with a given significance level (alpha), 
# numerator degrees of freedom (df1),
# and denominator degrees of freedom (df2), you can use the scipy.stats library's f.ppf function, 
# which gives the percent-point function (inverse of the cumulative distribution function) for the F-distribution. 
# Here's a Python function that does this:
from scipy.stats import f

def get_critical_f_value(alpha, df1, df2):
    # Calculate the critical F-value for a two-tailed test
    critical_f_value = f.ppf(1 - alpha / 2, df1, df2)
    
    return critical_f_value

# Example usage:
alpha = 0.05
df1 = 3  # Numerator degrees of freedom
df2 = 20  # Denominator degrees of freedom

critical_f = get_critical_f_value(alpha, df1, df2)
print("Critical F-Value:", critical_f)
# In this code, alpha represents the significance level, df1 is the degrees of freedom for the numerator, 
# and df2 is the degrees of freedom for the denominator.
# The function get_critical_f_value uses f.ppf to calculate
# the critical F-value for a two-tailed test at the specified significance level. You can replace alpha, 
# df1, and df2 with your own values to compute the critical F-value for your specific test.
# Ensure you have the scipy library installed, which includes the F-distribution functions.
# You can install it using pip if needed:


# In[ ]:





# Q3. Write a Python program that generates random samples from two normal distributions with known 
# variances and uses an F-test to determine if the variances are equal. The program should output the Fvalue, degrees of freedom, and p-value for the test

# In[3]:


# You can create a Python program to generate random samples from two normal distributions, perform an F-test to determine
# if their variances are equal, and output the F-value, degrees of freedom, and p-value for the test.
# Here's a sample program using the numpy and scipy.stats libraries:

def perform_variance_ratio_test(sample1, sample2):
    # Calculate the variances of the two samples
    var1 = np.var(sample1, ddof=1)  # ddof=1 for unbiased estimator
    var2 = np.var(sample2, ddof=1)
    
    # Calculate degrees of freedom
    df1 = len(sample1) - 1
    df2 = len(sample2) - 1
    
    # Calculate the F-statistic
    f_statistic = var1 / var2 if var1 >= var2 else var2 / var1
    
    # Calculate the p-value
    p_value = 2 * min(f.cdf(f_statistic, df1, df2), 1 - f.cdf(f_statistic, df1, df2))
    
    return f_statistic, df1, df2, p_value

# Generate random samples from normal distributions with known variances
np.random.seed(42)  # Seed for reproducibility
sample_size = 30  # Size of each sample
mean1, mean2 = 0, 0  # Means of the two distributions
variance1, variance2 = 1, 2  # Known variances of the two distributions

sample1 = np.random.normal(mean1, np.sqrt(variance1), sample_size)
sample2 = np.random.normal(mean2, np.sqrt(variance2), sample_size)

# Perform the variance ratio test
f_stat, df1, df2, p_val = perform_variance_ratio_test(sample1, sample2)

# Output the results
print("F-Value:", f_stat)
print("Degrees of Freedom (DF1, DF2):", df1, df2)
print("P-Value:", p_val)

# Interpretation
alpha = 0.05
if p_val < alpha:
    print("Reject the null hypothesis: Variances are not equal.")
else:
    print("Fail to reject the null hypothesis: Variances are equal.")
# In this code:
# We generate random samples from two normal distributions with known variances using numpy.random.normal.
# We calculate the sample variances and degrees of freedom.
# We calculate the F-statistic and the p-value using the perform_variance_ratio_test function, which performs the F-test.
# We output the F-value, degrees of freedom, and p-value.
# Finally, we interpret the results by comparing the p-value to the significance level (alpha) and 
# decide whether to reject or fail to reject the null hypothesis that the variances are equal.


# In[ ]:





# Q4.The variances of two populations are known to be 10 and 15. A sample of 12 observations is taken from 
# each population. Conduct an F-test at the 5% significance level to determine if the variances are 
# significantly different.

# In[4]:


# To conduct an F-test at the 5% significance level to determine if the variances of two populations
# are significantly different, you can follow these steps:
# Define your null and alternative hypotheses:
# Null Hypothesis (H0): The variances of the two populations are equal (σ1^2 = σ2^2).
# Alternative Hypothesis (Ha): The variances of the two populations are not equal (σ1^2 ≠ σ2^2).
# Choose a significance level (α) of 0.05, which corresponds to a 95% confidence level.
# Calculate the F-statistic using the formula:
# F = (s1^2 / s2^2)
# Where:
# s1^2 and s2^2 are the sample variances of the two populations.
# s1^2 = 10 (known variance of the first population)
# s2^2 = 15 (known variance of the second population)
# Calculate the degrees of freedom for the numerator (df1) and denominator (df2) of the F-distribution:
# df1 = n1 - 1 = 12 - 1 = 11
# df2 = n2 - 1 = 12 - 1 = 11
# Calculate the critical F-value for a two-tailed test at the 5% significance level with df1 and df2 degrees of freedom.
# You can use the scipy.stats library for this purpose, as shown in a previous answer.
# Compare the calculated F-statistic to the critical F-value:
# If F-statistic > Critical F-value, reject the null hypothesis.
# If F-statistic <= Critical F-value, fail to reject the null hypothesis.
# Let's calculate the critical F-value and conduct the test:
# Known variances
variance1 = 10
variance2 = 15

# Sample sizes
n1 = 12
n2 = 12

# Calculate the F-statistic
f_statistic = variance1 / variance2

# Degrees of freedom
df1 = n1 - 1
df2 = n2 - 1

# Calculate the critical F-value at the 5% significance level for a two-tailed test
alpha = 0.05
critical_f_value = f.ppf(1 - alpha / 2, df1, df2)

# Compare the F-statistic to the critical F-value
if f_statistic > critical_f_value:
    print("Reject the null hypothesis: Variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: Variances are not significantly different.")
# In this case, you would compare the calculated F-statistic to the critical F-value.
# If the F-statistic is greater than the critical F-value,
# you would reject the null hypothesis and conclude that the variances are significantly different.
# Otherwise, you would fail to reject the null hypothesis.


# In[ ]:





# Q5. A manufacturer claims that the variance of the diameter of a certain product is 0.005. A sample of 25 
# products is taken, and the sample variance is found to be 0.006. Conduct an F-test at the 1% significance 
# level to determine if the claim is justified.

# In[7]:


import scipy.stats as stats

# To conduct an F-test at the 1% significance level to determine if the manufacturer's claim about the variance of the product diameter is justified, follow these steps:

# Formulate Hypotheses:

# Null Hypothesis (H0): The manufacturer's claim is justified; the population variance is 0.005 (σ^2 = 0.005).
# Alternative Hypothesis (Ha): The manufacturer's claim is not justified; the population variance is different from 0.005 (σ^2 ≠ 0.005).
# Choose Significance Level (α): Set the significance level to α = 0.01 (1%).

# Calculate the Test Statistic:

# Calculate the F-statistic using the formula:

# �
# =
# �
# 1
# 2
# �
# 2
# 2
# F= 
# S 
# 2
# 2
# ​
 
# S 
# 1
# 2
# ​
 
# ​
 

# Where:

# �
# 1
# 2
# S 
# 1
# 2
# ​
#   is the sample variance (0.006).
# �
# 2
# 2
# S 
# 2
# 2
# ​
#   is the claimed population variance (0.005).
# Determine Degrees of Freedom:

# Degrees of freedom for the numerator (
# �
# �
# 1
# df 
# 1
# ​
#  ) is equal to the sample size minus 1, which is 
# �
# �
# 1
# =
# 25
# −
# 1
# =
# 24
# df 
# 1
# ​
#  =25−1=24.

# Degrees of freedom for the denominator (
# �
# �
# 2
# df 
# 2
# ​
#  ) is 1 because you are comparing to the claimed variance.

# Find Critical F-Value:

# Find the critical F-value(s) at the 1% significance level (α/2 = 0.005) with 
# �
# �
# 1
# df 
# 1
# ​
#   and 
# �
# �
# 2
# df 
# 2
# ​
#   degrees of freedom. You can use statistical tables or a calculator for this.

# Perform the Test:

# Compare the calculated F-statistic to the critical F-value(s).

# If 
# �
# F is outside the range of critical F-values, reject the null hypothesis.
# If 
# �
# F falls within the range, fail to reject the null hypothesis.
# Let's calculate the critical F-values and perform the test:
# Given data
sample_variance = 0.006
claimed_variance = 0.005
sample_size = 25
# Calculate the F-statistic
F_statistic = sample_variance / claimed_variance

# Degrees of freedom
df1 = sample_size - 1
df2 = 1  # Since you are comparing to the claimed variance

# Significance level
alpha = 0.01

# Find the critical F-values for a two-tailed test
critical_f_lower = stats.f.ppf(alpha / 2, df1, df2)
critical_f_upper = stats.f.ppf(1 - alpha / 2, df1, df2)

# Perform the test
if F_statistic < critical_f_lower or F_statistic > critical_f_upper:
    print("Reject the null hypothesis: The claim is not justified.")
else:
    print("Fail to reject the null hypothesis: The claim is justified.")
# In this code, we compare the calculated F-statistic to the critical F-values at the 1% significance level.
# If the F-statistic falls outside the range of critical F-values, we reject the null hypothesis, 
# indicating that the manufacturer's claim is not justified. Otherwise,
# we fail to reject the null hypothesis, suggesting that the claim is justified.


# In[ ]:





# Q6. Write a Python function that takes in the degrees of freedom for the numerator and denominator of an 
# F-distribution and calculates the mean and variance of the distribution. The function should return the 
# mean and variance as a tuple.
# 

# In[8]:


# You can calculate the mean and variance of an F-distribution based on its degrees of freedom for the 
# numerator (df1) and denominator (df2). Here's a Python function that does this:
def calculate_f_distribution_mean_variance(df1, df2):
    # Check that degrees of freedom are valid (df1 and df2 must be greater than 0)
    if df1 <= 0 or df2 <= 0:
        raise ValueError("Degrees of freedom must be greater than 0.")
    
    # Calculate the mean of the F-distribution
    if df2 > 2:
        mean = df2 / (df2 - 2)
    else:
        mean = float('inf')  # Mean is undefined when df2 <= 2
    
    # Calculate the variance of the F-distribution
    if df2 > 4:
        variance = (2 * (df2**2) * (df1 + df2 - 2)) / (df1 * (df2 - 2)**2 * (df2 - 4))
    else:
        variance = float('inf')  # Variance is undefined when df2 <= 4
    
    return mean, variance

# Example usage:
df1 = 5  # Degrees of freedom for the numerator
df2 = 10  # Degrees of freedom for the denominator

mean, variance = calculate_f_distribution_mean_variance(df1, df2)
print("Mean:", mean)
print("Variance:", variance)
# In this code:
# We first check if the degrees of freedom (df1 and df2) are valid (greater than 0).
# Then, we calculate the mean and variance of the F-distribution based on the provided degrees of freedom.
# The mean and variance are returned as a tuple.
# Please note that the mean and variance of the F-distribution have specific formulas depending on the degrees of freedom, and
# they can be undefined for certain values of df2. The code includes checks to handle these cases appropriately.


# In[ ]:





# Q7. A random sample of 10 measurements is taken from a normal population with unknown variance. The 
# sample variance is found to be 25. Another random sample of 15 measurements is taken from another 
# normal population with unknown variance, and the sample variance is found to be 20. Conduct an F-test 
# at the 10% significance level to determine if the variances are significantly different.

# In[9]:


# To conduct an F-test at the 10% significance level to determine if the variances of two populations are significantly different, follow these steps:

# Formulate Hypotheses:

# Null Hypothesis (H0): The variances of the two populations are equal (
# �
# 1
# 2
# =
# �
# 2
# 2
# σ 
# 1
# 2
# ​
#  =σ 
# 2
# 2
# ​
#  ).
# Alternative Hypothesis (Ha): The variances of the two populations are not equal (
# �
# 1
# 2
# ≠
# �
# 2
# 2
# σ 
# 1
# 2
# ​
 
# 
# =σ 
# 2
# 2
# ​
#  ).
# Choose Significance Level (α): Set the significance level to α = 0.10 (10%).

# Calculate the Test Statistic:

# Calculate the F-statistic using the formula:

# �
# =
# �
# 1
# 2
# �
# 2
# 2
# F= 
# S 
# 2
# 2
# ​
 
# S 
# 1
# 2
# ​
 
# ​
 

# Where:

# �
# 1
# 2
# S 
# 1
# 2
# ​
#   is the sample variance for the first population (25 for the sample of size 10).
# �
# 2
# 2
# S 
# 2
# 2
# ​
#   is the sample variance for the second population (20 for the sample of size 15).
# Determine Degrees of Freedom:

# Degrees of freedom for the numerator (
# �
# �
# 1
# df 
# 1
# ​
#  ) is 9 (sample size for the first population minus 1).

# Degrees of freedom for the denominator (
# �
# �
# 2
# df 
# 2
# ​
#  ) is 14 (sample size for the second population minus 1).

# Find Critical F-Value:

# Find the critical F-value for a two-tailed test at the 10% significance level (α/2 = 0.05) with 
# �
# �
# 1
# df 
# 1
# ​
#   and 
# �
# �
# 2
# df 
# 2
# ​
#   degrees of freedom. You can use statistical tables or a calculator for this.

# Perform the Test:

# Compare the calculated F-statistic to the critical F-value.

# If 
# �
# F is outside the range of critical F-values, reject the null hypothesis.
# If 
# �
# F falls within the range, fail to reject the null hypothesis.
# Let's calculate the critical F-value and perform the test:
# Given sample variances
sample_variance1 = 25
sample_variance2 = 20

# Sample sizes
n1 = 10
n2 = 15

# Calculate the F-statistic
F_statistic = sample_variance1 / sample_variance2

# Degrees of freedom
df1 = n1 - 1
df2 = n2 - 1

# Significance level
alpha = 0.10

# Find the critical F-values for a two-tailed test
critical_f_lower = stats.f.ppf(alpha / 2, df1, df2)
critical_f_upper = stats.f.ppf(1 - alpha / 2, df1, df2)

# Perform the test
if F_statistic < critical_f_lower or F_statistic > critical_f_upper:
    print("Reject the null hypothesis: The variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: The variances are not significantly different.")
# In this code, we compare the calculated F-statistic to the critical F-values at the 10% significance level.
# If the F-statistic falls outside the range of critical F-values, we reject the null hypothesis,
# indicating that the variances are significantly different.
# Otherwise, we fail to reject the null hypothesis.


# In[ ]:





# Q8. The following data represent the waiting times in minutes at two different restaurants on a Saturday 
# night: Restaurant A: 24, 25, 28, 23, 22, 20, 27; Restaurant B: 31, 33, 35, 30, 32, 36. Conduct an F-test at the 5% 
# significance level to determine if the variances are significantly different.

# In[10]:


# To conduct an F-test at the 5% significance level to determine if the variances of the waiting times at two different restaurants (Restaurant A and Restaurant B) are significantly different, follow these steps:

# Formulate Hypotheses:

# Null Hypothesis (H0): The variances of the waiting times at Restaurant A and Restaurant B are equal (
# �
# �
# 2
# =
# �
# �
# 2
# σ 
# A
# 2
# ​
#  =σ 
# B
# 2
# ​
#  ).
# Alternative Hypothesis (Ha): The variances of the waiting times at Restaurant A and Restaurant B are not equal (
# �
# �
# 2
# ≠
# �
# �
# 2
# σ 
# A
# 2
# ​
 
# 
# =σ 
# B
# 2
# ​
#  ).
# Choose Significance Level (α): Set the significance level to α = 0.05 (5%).

# Calculate the Test Statistic:

# Calculate the F-statistic using the formula:

# �
# =
# �
# �
# 2
# �
# �
# 2
# F= 
# S 
# B
# 2
# ​
 
# S 
# A
# 2
# ​
 
# ​
 

# Where:

# �
# �
# 2
# S 
# A
# 2
# ​
#   is the sample variance of waiting times at Restaurant A.
# �
# �
# 2
# S 
# B
# 2
# ​
#   is the sample variance of waiting times at Restaurant B.
# Determine Degrees of Freedom:

# Degrees of freedom for the numerator (
# �
# �
# 1
# df 
# 1
# ​
#  ) is one less than the number of data points in Restaurant A.

# �
# �
# 1
# =
# �
# �
# −
# 1
# df 
# 1
# ​
#  =n 
# A
# ​
#  −1
# Degrees of freedom for the denominator (
# �
# �
# 2
# df 
# 2
# ​
#  ) is one less than the number of data points in Restaurant B.

# �
# �
# 2
# =
# �
# �
# −
# 1
# df 
# 2
# ​
#  =n 
# B
# ​
#  −1
# Find Critical F-Value:

# Find the critical F-value for a two-tailed test at the 5% significance level (α/2 = 0.025) with 
# �
# �
# 1
# df 
# 1
# ​
#   and 
# �
# �
# 2
# df 
# 2
# ​
#   degrees of freedom. You can use statistical tables or a calculator for this.

# Perform the Test:

# Compare the calculated F-statistic to the critical F-value.

# If 
# �
# F is outside the range of critical F-values, reject the null hypothesis.
# If 
# �
# F falls within the range, fail to reject the null hypothesis.
# Let's calculate the critical F-value and perform the test:
# Data for waiting times at Restaurant A and Restaurant B
data_A = np.array([24, 25, 28, 23, 22, 20, 27])
data_B = np.array([31, 33, 35, 30, 32, 36])

# Calculate the sample variances
variance_A = np.var(data_A, ddof=1)
variance_B = np.var(data_B, ddof=1)

# Calculate the F-statistic
F_statistic = variance_A / variance_B

# Degrees of freedom
df1 = len(data_A) - 1
df2 = len(data_B) - 1

# Significance level
alpha = 0.05

# Find the critical F-values for a two-tailed test
critical_f_lower = stats.f.ppf(alpha / 2, df1, df2)
critical_f_upper = stats.f.ppf(1 - alpha / 2, df1, df2)

# Perform the test
if F_statistic < critical_f_lower or F_statistic > critical_f_upper:
    print("Reject the null hypothesis: The variances are significantly different.")
else:
    print("Fail to reject the null hypothesis: The variances are not significantly different.")
# In this code, we calculate the F-statistic, degrees of freedom, and critical F-values to perform the F-test.
# If the F-statistic falls outside the range of critical F-values, we reject the null hypothesis, 
# indicating that the variances are significantly different.
# Otherwise, we fail to reject the null hypothesis.


# In[ ]:





# Q9. The following data represent the test scores of two groups of students: Group A: 80, 85, 90, 92, 87, 83; 
# Group B: 75, 78, 82, 79, 81, 84. Conduct an F-test at the 1% significance level to determine if the variances 
# are significantly different.
 To conduct an F-test to determine if the variances of two groups are significantly different, you can follow these steps:

Step 1: State the null and alternative hypotheses:

Null Hypothesis (H0): The variances of Group A and Group B are equal.
Alternative Hypothesis (Ha): The variances of Group A and Group B are not equal.

Step 2: Calculate the sample variances for each group:

For Group A:
Sample Size (n1) = 6
Sample Mean (x̄1) = (80 + 85 + 90 + 92 + 87 + 83) / 6 = 517 / 6 ≈ 86.17
Sample Variance (s1^2) = [(80 - 86.17)^2 + (85 - 86.17)^2 + (90 - 86.17)^2 + (92 - 86.17)^2 + (87 - 86.17)^2 + (83 - 86.17)^2] / (6 - 1) ≈ 20.95

For Group B:
Sample Size (n2) = 6
Sample Mean (x̄2) = (75 + 78 + 82 + 79 + 81 + 84) / 6 = 479 / 6 ≈ 79.83
Sample Variance (s2^2) = [(75 - 79.83)^2 + (78 - 79.83)^2 + (82 - 79.83)^2 + (79 - 79.83)^2 + (81 - 79.83)^2 + (84 - 79.83)^2] / (6 - 1) ≈ 6.64

Step 3: Calculate the F-statistic:

F = (s1^2) / (s2^2)

F = 20.95 / 6.64 ≈ 3.16

Step 4: Determine the critical F-value from the F-distribution table for a 1% significance level and degrees of freedom (df1 = n1 - 1 = 6 - 1 = 5 and df2 = n2 - 1 = 6 - 1 = 5). You can use an F-table or an online calculator to find the critical F-value.

At a 1% significance level and degrees of freedom (df1 = 5, df2 = 5), the critical F-value is approximately 11.07.

Step 5: Compare the calculated F-statistic with the critical F-value:

3.16 < 11.07

Step 6: Make a decision:

Since the calculated F-statistic (3.16) is less than the critical F-value (11.07), we do not reject the null hypothesis.

Step 7: Draw a conclusion:

At the 1% significance level, there is not enough evidence to conclude that the variances of Group A and Group B are significantly different. In other words, it is reasonable to assume that the variances are equal for the two groups.





# In[ ]:





# #  <P style="color:GREEN"> THNAK YOU, THAT'S ALL </p>
