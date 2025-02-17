# Customer Segmentation Analysis using K-Means Clustering

## Project Overview
This project analyzes customer data to identify distinct consumer segments using clustering techniques (K-Means) and PCA for visualization. The goal is to derive insights for personalized marketing strategies based on customer spending behavior, demographics, and purchasing habits.

## Dataset Description
**Source File:** `marketing_campaign.csv`  
**Separator Used:** `;` (semicolon)

### Key Features:
- **Demographics:** `Year_Birth` (birth year), `Income`, `Kidhome`, `Teenhome`, `Age`
- **Customer Engagement:** `Recency`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`
- **Spending Behavior:** `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- **Marketing Campaign Responses:** `AcceptedCmp1` - `AcceptedCmp5`, `Response`
- **Date Feature:** `Dt_Customer` (converted to datetime format)
- **Dropped Columns:** `Z_CostContact`, `Z_Revenue` (constant values)

## Preprocessing Steps
### Data Cleaning:
- Converted `Dt_Customer` to datetime format.
- Filled missing values in `Income` with the median.
- Encoded categorical variables (`Education`, `Marital_Status`).
- Created an `Age` column from `Year_Birth`.
- Dropped irrelevant columns (`Z_CostContact`, `Z_Revenue`).

## Exploratory Data Analysis (EDA)
- **Correlation Heatmap:** Identified relationships between numerical features.
- **Age Distribution:** Visualized customer age spread.
- **Income Distribution:** Analyzed income distribution among customers.
- **Spending Behavior:** Examined total spending across different product categories.

## Clustering Analysis
- **Feature Selection:** Used key numerical attributes for clustering.
- **Standardization:** Applied `StandardScaler` for normalization.
- **Elbow Method:** Determined the optimal number of clusters.
- **K-Means Clustering:** Implemented with `k=4` (optimal from Elbow Method).
- **PCA Visualization:** Reduced dimensions to 2D for visualizing clusters.

## Key Outputs
- **Summary Statistics:** Descriptive analysis of customer attributes.
- **Heatmaps & Distribution Plots:** Visual insights into customer demographics & spending.
- **Elbow Method Plot:** Identified the best number of clusters.
- **Cluster Segmentation:** Segregated customers into meaningful segments.
- **Cluster-Wise Statistics:** Displayed average values for each cluster.

## Usage Instructions
### Extract Data:
1. Ensure the dataset (`marketing_campaign.csv`) is placed in the correct directory.
2. Run the script to load and preprocess the data.

### Run the Analysis:
1. The script will generate EDA visualizations, clustering analysis, and customer segments.
2. Cluster-wise statistics will be displayed for further business insights.

### Interpretation:
- Use segmentation results to tailor marketing campaigns, promotions, and product recommendations.
- Identify high-value customers and budget-conscious shoppers for targeted engagement.

## Next Steps & Recommendations
- **Personalized Marketing:** Leverage insights to create customized offers per cluster.
- **Customer Retention Strategies:** Develop loyalty programs based on spending behavior.
- **Further Analysis:** Extend segmentation with additional clustering algorithms (e.g., Hierarchical, DBSCAN).
