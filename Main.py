import zipfile
import os

# Define the path to the uploaded file
zip_file_path = "/mnt/data/archive (1).zip"
extract_folder = "/mnt/data/customer_personality_analysis"

# Extract the contents of the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# List the extracted files
extracted_files = os.listdir(extract_folder)
Extracted_files

import pandas as pd

# Load the dataset
file_path = os.path.join(extract_folder, "marketing_campaign.csv")
df = pd.read_csv(file_path, sep=";")  # Using semicolon as the separator

# Display basic information and the first few rows
df.info(), df.head()

# Reload the dataset with tab-separated values
df = pd.read_csv(file_path, delimiter="\t")

# Display basic information and the first few rows
df.info(), df.head()

# Convert 'Dt_Customer' to datetime format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# Handle missing values in 'Income' by replacing with median income
df['Income'].fillna(df['Income'].median(), inplace=True)

# Encode categorical variables
df['Education'] = df['Education'].astype('category')
df['Marital_Status'] = df['Marital_Status'].astype('category')

# Create Age column from Year_Birth
df['Age'] = 2024 - df['Year_Birth']

# Drop unnecessary columns 'Z_CostContact' and 'Z_Revenue' as they are constant values
df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)

# Display cleaned data info and the first few rows
df.info(), df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use("ggplot")

# Summary statistics
summary_stats = df.describe()

# Display statistical summary
import ace_tools as tools
tools.display_dataframe_to_user(name="Summary Statistics", dataframe=summary_stats)

# Correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Customer Attributes")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Income'], bins=20, kde=True)
plt.title("Income Distribution of Customers")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# Spending habits across product categories
spending_columns = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df[spending_columns].sum().plot(kind="bar", figsize=(10,5), color="skyblue")
plt.title("Total Spending by Product Category")
plt.ylabel("Total Amount Spent")
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Selecting relevant numerical features for clustering
clustering_features = ["Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", 
                       "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds", 
                       "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", 
                       "NumWebVisitsMonth", "Age"]

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[clustering_features])

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia (Within-cluster Sum of Squares)")
plt.title("Elbow Method for Optimal Cluster Selection")
plt.show()

# Apply KMeans clustering with optimal clusters (assume k=4 based on Elbow method)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Visualizing clusters using PCA (reducing dimensions to 2D)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)
df["PCA1"] = reduced_features[:, 0]
df["PCA2"] = reduced_features[:, 1]

# Scatter plot for clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster"], palette="tab10", alpha=0.7)
plt.title("Customer Segmentation (K-Means Clustering)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

# Display cluster-wise statistics
cluster_summary = df.groupby("Cluster")[clustering_features].mean()
tools.display_dataframe_to_user(name="Cluster-wise Statistics", dataframe=cluster_summary)

Output












