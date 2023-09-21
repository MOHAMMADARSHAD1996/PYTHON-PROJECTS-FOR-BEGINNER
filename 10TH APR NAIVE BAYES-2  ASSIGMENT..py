#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> NAIVE BAYES-2   </p>

# Q1. A company conducted a survey of its employees and found that 70% of the employees use the 
# company's health insurance plan, while 40% of the employees who use the plan are smokers. What is the 
# probability that an employee is a smoker given that he/she uses the health insurance plan?
To find the probability that an employee is a smoker given that he/she uses the health insurance plan, you can use conditional probability. This can be calculated using the formula for conditional probability:

\[P(A | B) = \frac{P(A \cap B)}{P(B)}\]

In this case:
- A represents the event of being a smoker.
- B represents the event of using the health insurance plan.

You are given:
- \(P(B)\), the probability that an employee uses the health insurance plan, which is 70% or 0.70.
- \(P(A | B)\), the probability of being a smoker given that the employee uses the health insurance plan, which you want to find.
- \(P(A)\), the probability of being a smoker, which you need to calculate.

To calculate \(P(A)\), you can use the information that 40% of the employees who use the plan are smokers. So:

\[P(A) = 0.40\]

Now, you can plug these values into the formula to find \(P(A | B)\):

\[P(A | B) = \frac{P(A \cap B)}{P(B)}\]
\[P(A | B) = \frac{0.40}{0.70}\]

Now, calculate \(P(A | B)\):

\[P(A | B) = \frac{0.40}{0.70} \approx 0.5714\]

So, the probability that an employee is a smoker given that he/she uses the health insurance plan is approximately 0.5714 or 57.14%.
# In[ ]:





# Q2. What is the difference between Bernoulli Naive Bayes and Multinomial Naive Bayes?
Bernoulli Naive Bayes and Multinomial Naive Bayes are two different variants of the Naive Bayes classification algorithm, and they are commonly used in text classification and other machine learning tasks. The main difference between them lies in the type of data they are designed to handle and the assumptions they make about the data:

1. **Bernoulli Naive Bayes:**
   - **Data Type:** Bernoulli Naive Bayes is primarily used for binary or boolean data, where each feature represents the presence or absence of a particular attribute.
   - **Assumption:** It assumes that the features are binary (0 or 1) and independent of each other.
   - **Use Cases:** It is commonly used for text classification tasks like spam detection, sentiment analysis, and document classification, where the focus is on the presence or absence of specific words or features in a document.

2. **Multinomial Naive Bayes:**
   - **Data Type:** Multinomial Naive Bayes is used for discrete data, typically in the form of counts or frequencies. It is well-suited for text data where features represent word counts or term frequencies.
   - **Assumption:** It assumes that the features follow a multinomial distribution (i.e., the counts of features) and that they are independent of each other, given the class label.
   - **Use Cases:** It is widely used in natural language processing (NLP) tasks such as text classification, topic modeling, and document categorization, where the focus is on word frequencies and counts within documents.

In summary, the choice between Bernoulli Naive Bayes and Multinomial Naive Bayes depends on the nature of your data. If you are dealing with binary data or data that can be represented as binary (e.g., presence or absence of words), Bernoulli Naive Bayes may be more appropriate. On the other hand, if your data consists of counts or frequencies (e.g., word counts in documents), Multinomial Naive Bayes is often a better choice.
# In[ ]:





# Q3. How does Bernoulli Naive Bayes handle missing values?
Bernoulli Naive Bayes, like other variants of the Naive Bayes algorithm, handles missing values differently depending on how the model is trained and how the missing values are represented during training and prediction. Here are a few common approaches to handling missing values with Bernoulli Naive Bayes:

1. **Ignoring Missing Values (Dropping Rows):** One straightforward approach is to simply ignore rows (samples) with missing values during both training and prediction. This means that any data point with a missing value in any of its features is excluded from the analysis. While this is a simple approach, it can result in a loss of valuable information, especially if you have a substantial amount of missing data.

2. **Treating Missing Values as a Separate Category:** Another option is to treat missing values as a separate category or state for each feature. In the context of Bernoulli Naive Bayes, this would involve considering an additional binary variable for each feature, indicating whether the value is missing or not. This approach can help retain information about the missingness pattern.

3. **Imputing Missing Values:** Imputation involves filling in missing values with estimated or imputed values. For Bernoulli Naive Bayes, you might impute missing values in binary features by assigning a default value (e.g., 0 or 1) or using a more sophisticated imputation method, such as probabilistic imputation based on the distribution of the feature.

4. **Using a Special Token (Text Data):** In the case of text classification with Bernoulli Naive Bayes, if you are dealing with missing values in a text feature (e.g., a word is missing), you can treat it as a special "missing" token or simply ignore it, depending on your dataset and the nature of the problem.

The choice of how to handle missing values in Bernoulli Naive Bayes depends on the nature of your data and the specific problem you are trying to solve. It's essential to consider the impact of missing data on your model's performance and choose an approach that makes the most sense for your application. Additionally, it's a good practice to analyze the reasons for missing data and determine whether they are missing completely at random, missing at random, or missing not at random, as this can influence your choice of handling method.
# In[ ]:





# Q4. Can Gaussian Naive Bayes be used for multi-class classification?
Yes, Gaussian Naive Bayes can be used for multi-class classification. Gaussian Naive Bayes is an extension of the Naive Bayes algorithm that is suitable for continuous or real-valued features. While it's often associated with binary or two-class classification problems, it can be adapted for multi-class classification as well.

In the context of multi-class classification, Gaussian Naive Bayes operates as follows:

1. **Modeling the Data:** For each class in the multi-class problem, Gaussian Naive Bayes estimates the mean and variance of the feature values for that class. It assumes that the feature values for each class are normally distributed (follow a Gaussian distribution).

2. **Predicting the Class:** When making predictions, Gaussian Naive Bayes calculates the likelihood of the observed feature values belonging to each class based on the estimated means and variances. It combines these likelihoods with the prior probabilities of each class (the proportion of each class in the training data) and applies Bayes' theorem to calculate the posterior probabilities for each class. The class with the highest posterior probability is then predicted as the final class label.

3. **Handling Multiple Classes:** To handle multiple classes, the algorithm repeats the above process for each class, comparing the posterior probabilities to determine the most likely class for a given set of feature values.

While Gaussian Naive Bayes can be used for multi-class classification, it makes the assumption that the features are normally distributed within each class, which may not always hold true in practice. In cases where this assumption is significantly violated, other algorithms or variations of Naive Bayes (such as Multinomial Naive Bayes for discrete features) may be more appropriate.

In summary, Gaussian Naive Bayes can be extended to handle multi-class classification by applying the basic principles of the Naive Bayes algorithm to multiple classes, but its effectiveness depends on the underlying distribution of the data and the appropriateness of the Gaussian assumption for the features.
# In[ ]:





# Q5. Assignment:
# Data preparation:
# 
# Download the "Spambase Data Set" from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/
# datasets/Spambase). This dataset contains email messages, where the goal is to predict whether a message 
# is spam or not based on several input features.
# 
# 
# Implementation:
# 
# Implement Bernoulli Naive Bayes, Multinomial Naive Bayes, and Gaussian Naive Bayes classifiers using the 
# scikit-learn library in Python. Use 10-fold cross-validation to evaluate the performance of each classifier on the 
# dataset. You should use the default hyperparameters for each classifier.
# Results:
# 
# Report the following performance metrics for each classifier:
# 
# Accuracy
# 
# Precision
# 
# Recall
# 
# F1 score
# 
# 
# Discussion:
# 
# Discuss the results you obtained. Which variant of Naive Bayes performed the best? Why do you think that is 
# the case? Are there any limitations of Naive Bayes that you observed?
# 
# 
# Conclusion:
# 
# Summarise your findings and provide some suggestions for future work
 Here's a step-by-step guide on how you can perform this task:

**Data Preparation:**
1. Download the "Spambase Data Set" from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Spambase).

2. Load the dataset into your Python environment. You can use libraries like Pandas to read and manipulate the data.

3. Split the dataset into features (X) and the target variable (y), where X contains the input features, and y contains the binary labels (spam or not spam).

4. Normalize or preprocess the features if necessary. Since you are working with text data, you may need to convert it into numerical features, such as using techniques like TF-IDF.

**Implementation:**
5. Import the necessary modules from scikit-learn for each type of Naive Bayes classifier: `BernoulliNB`, `MultinomialNB`, and `GaussianNB`.

6. Create instances of each classifier.

7. Perform 10-fold cross-validation for each classifier. You can use the `cross_val_score` function from scikit-learn to simplify this process. Ensure that you evaluate each classifier using the desired performance metrics (accuracy, precision, recall, F1-score) within the cross-validation loop.

**Results:**
8. After running the cross-validation for each classifier, calculate the mean and standard deviation of the performance metrics (accuracy, precision, recall, F1-score) across the 10 folds.

9. Report the results for each classifier, including mean and standard deviation values for each performance metric.

**Discussion:**
10. Analyze the results. Compare the performance of the three Naive Bayes variants (Bernoulli, Multinomial, Gaussian) in terms of accuracy, precision, recall, and F1-score. Consider why one variant might have performed better than others.

11. Discuss any limitations or challenges you observed during the analysis. For example, Naive Bayes assumes independence between features, which might not hold for text data.

**Conclusion:**
12. Summarize your findings. State which variant of Naive Bayes performed the best on the Spambase dataset and why you think it performed better.

13. Provide suggestions for future work. You can discuss potential improvements, such as feature engineering, hyperparameter tuning, or trying different classification algorithms to improve performance further.

Remember to document your code, results, and findings thoroughly to make your analysis reproducible and understandable.
# In[ ]:





# 
# #  <P style="color:GREEN"> Thank You ,That's All </p>
