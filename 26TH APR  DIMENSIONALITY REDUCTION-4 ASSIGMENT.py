#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# #  <P style="color:purple"> DIMENSIONALITY REDUCTION-4 </p>

# Objective:
# The objective of this assignment is to implement PCA on a given dataset and analyse the results.
# Deliverables:
# Jupyter notebook containing the code for the PCA implementation.
# A report summarising the results of PCA and clustering analysis.
# Scatter plot showing the results of PCA.
# A table showing the performance metrics for the clustering algorithm.
# Additional Information:
# You can use the python programming language.
# You can use any other machine learning libraries or tools as necessary.
# You can use any visualisation libraries or tools as necessary.
# Instructions:
# Download the wine dataset from the UCI Machine Learning Repository 
# 

# Load the dataset into a Pandas dataframe.
To load the dataset into a Pandas DataFrame, you can use the pandas.read_csv() function if your data is in a CSV file. Assuming your data is in a CSV file named "wine_data.csv," you can do the following:
import pandas as pd

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("wine_data.csv")

# Display the first few rows of the DataFrame to verify the data loading
print(df.head())
Make sure to replace "wine_data.csv" with the actual file path or URL to your dataset. This code will read the CSV file and create a DataFrame named "df" with your data.

# In[ ]:





# Split the dataset into features and target variables.
To split the dataset into features and the target variable, you can use the following code:
# Assuming your dataset is already loaded into a DataFrame named 'df'

# Features (independent variables)
X = df.drop(columns=['class'])

# Target variable (dependent variable)
y = df['class']
In this code:

X contains all the feature columns (Alcohol, Malicacid, Ash, Alcalinity_of_ash, Magnesium, Total_phenols, Flavanoids, Nonflav_phenols, Proanthocyanins, Color_intensity, Hue, of_diluted_wines, Proline).
y contains the target variable 'class'.
Now, you can use the X and y variables for further data analysis, preprocessing, and modeling.
# In[ ]:





# Perform data preprocessing (e.g., scaling, normalisation, missing value imputation) as necessary
Data preprocessing is an essential step in preparing your dataset for machine learning. In this case, you've mentioned that there are no missing values, and the data types seem to be appropriate for their respective roles (e.g., 'class' is categorical, and the other variables are continuous or integer). However, you can perform feature scaling or normalization to ensure that the features are on similar scales, which can improve the performance of some machine learning algorithms. I'll show you how to perform feature scaling using the StandardScaler from Scikit-Learn:
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the feature columns (except the target variable)
X_scaled = scaler.fit_transform(X)

# The resulting X_scaled is a NumPy array with scaled feature values
Now, X_scaled contains your feature variables with standardized values. This is especially useful if you plan to use algorithms like Support Vector Machines (SVM) or k-Nearest Neighbors (k-NN), which are sensitive to feature scales.

Remember that the choice of preprocessing steps can depend on the specific machine learning algorithms you plan to use and the nature of your data. Additionally, it's a good practice to split your dataset into training and testing subsets to evaluate the performance of your models properly.
# In[ ]:





# Implement PCA on the preprocessed dataset using the scikit-learn library.
To implement Principal Component Analysis (PCA) on the preprocessed dataset using the scikit-learn library, you can follow these steps:

Import the necessary libraries.
Standardize the features (if you haven't already done so).
Apply PCA to reduce the dimensionality of the dataset.
Choose the number of principal components to keep based on explained variance.
Transform the data using the selected number of principal components.
Here's how you can do it:
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming you've already loaded and preprocessed your data (X_scaled)

# Initialize the PCA model
pca = PCA()

# Fit PCA to the standardized features
pca.fit(X_scaled)

# Determine the number of principal components to keep based on explained variance
# For example, you can set a threshold for the explained variance (e.g., 95%)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()
n_components = len(cumulative_explained_variance[cumulative_explained_variance <= 0.95])

# Initialize PCA again with the selected number of components
pca = PCA(n_components=n_components)

# Fit and transform the data to the selected number of principal components
X_pca = pca.fit_transform(X_scaled)

# X_pca contains the dataset transformed into a lower-dimensional space
In this code:
PCA() initializes the PCA model.
fit() computes the principal components from the standardized features.
explained_variance_ratio_ contains the variance explained by each component.
cumulative_explained_variance computes the cumulative explained variance.
n_components is determined based on a threshold (e.g., 95% explained variance) or another criterion.
The resulting X_pca contains the dataset with reduced dimensions based on the selected number of principal components. You can use this reduced dataset for further analysis or modeling.
# In[ ]:





# Determine the optimal number of principal components to retain based on the explained variance ratio.
o determine the optimal number of principal components to retain based on the explained variance ratio, you can plot the cumulative explained variance and visually inspect the point where it levels off or reaches a desired threshold. This will help you decide how many principal components are sufficient to retain most of the information in the data. Here's how you can do it:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming you've already loaded and preprocessed your data (X_scaled)

# Initialize the PCA model
pca = PCA()

# Fit PCA to the standardized features
pca.fit(X_scaled)

# Compute the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.grid(True)

# You can also add a horizontal line to represent a desired threshold
# For example, if you want to retain 95% of the variance, you can add a line at y=0.95

# Determine the number of components that explain at least 95% of the variance
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
print(f"Number of components to retain 95% variance: {n_components_95}")

plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()
In this code:
We calculate the cumulative explained variance by cumulatively summing the explained variance ratios.
We create a plot to visualize the cumulative explained variance as a function of the number of principal components.
We add a horizontal line at the desired threshold (e.g., 95% explained variance).
The point where the cumulative explained variance crosses the threshold or levels off is a good indication of the optimal number of principal components to retain.
You can adjust the threshold based on your specific requirements. In the code above, n_components_95 will contain the number of principal components needed to retain at least 95% of the variance.
# In[ ]:





# Visualise the results of PCA using a scatter plot.
To visualize the results of PCA using a scatter plot, you can create scatter plots with the first two or three principal components as the x and y axes. Since your dataset contains multiple features, this will help you see how the data clusters or separates in a reduced-dimensional space. Here's how you can do it using Python and Matplotlib:
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming you've already loaded and preprocessed your data (X_scaled)

# Initialize the PCA model with the desired number of components
n_components = 2  # You can also use 3 for a 3D scatter plot
pca = PCA(n_components=n_components)

# Fit and transform the data to the selected number of principal components
X_pca = pca.fit_transform(X_scaled)

# Create a scatter plot
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']  # You can specify colors for different classes

# Assuming 'class' is your target variable
for target_class, color in zip(df['class'].unique(), colors):
    plt.scatter(X_pca[df['class'] == target_class, 0], X_pca[df['class'] == target_class, 1],
                color=color, label=target_class, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()
In this code:
n_components specifies the number of principal components you want to use for the scatter plot (in this case, 2 for a 2D plot).
We loop through unique class labels (assuming 'class' is your target variable) and plot data points with different colors for each class.
You can customize the colors and labels based on your dataset.
This scatter plot will give you a visual representation of how your data is distributed in the reduced-dimensional space defined by the principal components. It can help you identify clusters or patterns in the data.
# In[ ]:





# Perform clustering on the PCA-transformed data using K-Means clustering algorithm.
To perform clustering on the PCA-transformed data using the K-Means clustering algorithm, you can follow these steps:

Import the necessary libraries.
Standardize and PCA-transform the data.
Choose the number of clusters (K) using methods like the Elbow Method or Silhouette Score.
Apply K-Means clustering.
Visualize the results.
Here's a Python code example using Scikit-Learn for clustering:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Assuming you've already loaded and preprocessed your data (X_scaled)

# Initialize the PCA model
n_components = 2  # Choose the number of principal components for visualization
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Standardize the PCA-transformed data (optional but recommended)
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca)

# Determine the optimal number of clusters (K)
# You can use the Elbow Method or Silhouette Score
# Let's use the Silhouette Score here as an example
silhouette_scores = []
for k in range(2, 11):  # You can adjust the range as needed
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca_scaled)
    silhouette_avg = silhouette_score(X_pca_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the Silhouette Scores to choose the optimal K
plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()

# Choose the optimal K based on the plot (e.g., where Silhouette Score is highest)
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 to account for the range starting at 2

# Apply K-Means clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca_scaled)

# Visualize the clustering results
plt.figure(figsize=(8, 6))
for cluster_label in range(optimal_k):
    plt.scatter(X_pca[cluster_labels == cluster_label, 0], X_pca[cluster_labels == cluster_label, 1],
                label=f'Cluster {cluster_label + 1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='black', marker='X', label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'K-Means Clustering (K={optimal_k})')
plt.legend()
plt.grid(True)
plt.show()
In this code:

We first perform PCA and standardize the transformed data (you can skip standardization if you prefer).
We then determine the optimal number of clusters (K) using the Silhouette Score, which measures the quality of clustering.
We visualize the Silhouette Scores to choose the optimal K.
Finally, we apply K-Means clustering with the optimal K and visualize the clustering results.
You can adjust the range of K values and other parameters as needed for your specific dataset
# In[ ]:





# Interpret the results of PCA and clustering analysis.
Interpreting the results of Principal Component Analysis (PCA) and clustering analysis involves understanding how the data has been transformed and how it has been grouped into clusters. Here's how you can interpret the results:

PCA Results Interpretation:

Principal Components: PCA reduces the dimensionality of the data by transforming the original features into a new set of uncorrelated variables called principal components (PCs). These PCs are linear combinations of the original features.

Explained Variance Ratio: The explained variance ratio for each PC indicates the proportion of variance in the data explained by that component. Higher values indicate that the PC captures more information from the original data. You typically look at the cumulative explained variance to decide how many components to retain.

Scatter Plot: The scatter plot of PCA-transformed data (often the first two principal components) provides insights into the structure of the data in the reduced-dimensional space. Here's what you can interpret from the scatter plot:

Clustering or Separation: Observe if data points form clusters or are spread out. Clusters may suggest the presence of distinct groups in the data.
Outliers: Identify any data points that are far from the main clusters. These may be outliers or anomalies.
Separation by Class: If you have class labels (e.g., 'class' in your dataset), see if the clusters correspond to different classes.
Clustering Analysis Results Interpretation:

Number of Clusters (K): Determine the optimal number of clusters based on a metric like the Silhouette Score or the Elbow Method. This tells you how many groups the data is best divided into.

Cluster Labels: After applying K-Means or another clustering algorithm, each data point is assigned to a cluster label. You can interpret the results as follows:

Cluster Centers: The centroids of the clusters represent the center of each group.
Cluster Size: Observe if the clusters have roughly equal sizes or if some are much larger or smaller.
Cluster Separation: Determine how distinct the clusters are from each other.
Visualization: Visualize the results using a scatter plot (as shown in the previous response). Interpret the clusters based on their spatial distribution in the reduced-dimensional space.

Domain Knowledge: Incorporate domain knowledge if available. If you have prior information about what the clusters might represent, you can use that knowledge to interpret the clusters more effectively.

Further Analysis: Depending on the goals of your analysis, you might perform additional tests or analysis within each cluster to understand the characteristics of each group.

Ultimately, the interpretation of PCA and clustering results depends on the specific context of your dataset and the goals of your analysis. It's important to keep in mind that while these techniques can reveal patterns and groupings in your data, they may not provide direct causal insights, and additional analysis may be needed to draw meaningful conclusions.
# In[ ]:





# #  <P style="color:green"> THAT'S ALL, THANK YOU    </p>
