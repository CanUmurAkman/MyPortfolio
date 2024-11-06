#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load the dataset
file_path = '/Users/macbookpro/Desktop/MFT Energy Case Study/messy_trading_data.csv'
df = pd.read_csv(file_path)


# In[3]:


# Output basic info about the dataset
print(df.info())


# In[4]:


# Display number of rows and columns
print(f"\nNumber of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


# In[5]:


# Check for NaN values
print(df.isna().sum())


# In[6]:


# Check for empty cells
print((df == '').sum())


# In[7]:


# Display the first few rows to inspect the data
print(df.head())


# In[8]:


# Remove rows where the Temperature column has NaN values
df_cleaned = df.dropna(subset=['Temperature'])


# In[9]:


# Display the first few rows to verify the data
print(df_cleaned.head())


# In[10]:


# Use .loc to modify the column without raising a warning
df_cleaned.loc[:, 'Date'] = pd.to_datetime(df_cleaned['Date'])


# In[11]:


# Sort the DataFrame by the date column in ascending order
df_cleaned = df_cleaned.sort_values(by='Date', ascending=True)


# In[12]:


# Reset the index
df_cleaned.reset_index(drop=True, inplace=True)


# In[13]:


# Count the number of negative values and text values in the PRICE column
negative_price_count = (df_cleaned['Price'] < 0).sum()
text_price_count = df_cleaned['Price'].apply(lambda x: isinstance(x, str)).sum()

print(f"Negative values in Price column: {negative_price_count}")
print(f"Text values in Price column: {text_price_count}")


# In[14]:


# Count the number of negative values and text values in the VOLUME column
negative_volume_count = (df_cleaned['Volume'] < 0).sum()
text_volume_count = df_cleaned['Volume'].apply(lambda x: isinstance(x, str)).sum()

print(f"Negative values in Volume column: {negative_volume_count}")
print(f"Text values in Volume column: {text_volume_count}")


# In[15]:


# Remove rows where Price or Volume has negative values
df_cleaned = df_cleaned[(df_cleaned['Price'] >= 0) & (df_cleaned['Volume'] >= 0)]


# In[16]:


# Check for values containing "," in Price, Volume, and Temperature columns
comma_price_count = df_cleaned['Price'].astype(str).str.contains(',').sum()
comma_volume_count = df_cleaned['Volume'].astype(str).str.contains(',').sum()
comma_temperature_count = df_cleaned['Temperature'].astype(str).str.contains(',').sum()

print(f"Number of values containing ',' in Price: {comma_price_count}")
print(f"Number of values containing ',' in Volume: {comma_volume_count}")
print(f"Number of values containing ',' in Temperature: {comma_temperature_count}")


# In[17]:


# Ensure Price, Volume, and Temperature columns are in numeric format
# If there are any non-numeric values, they will be converted to NaN
df_cleaned['Price'] = pd.to_numeric(df_cleaned['Price'], errors='coerce')
df_cleaned['Volume'] = pd.to_numeric(df_cleaned['Volume'], errors='coerce')
df_cleaned['Temperature'] = pd.to_numeric(df_cleaned['Temperature'], errors='coerce')


# In[18]:


# Display any rows with NaNs (to inspect if type conversion introduced NaNs)
print("\nRows with NaNs after conversion (if any):")
print(df_cleaned[df_cleaned.isna().any(axis=1)])


# In[19]:


# Count the number of duplicate values in the Date column
duplicate_count = df_cleaned.duplicated(subset=['Date']).sum()

print(f"Number of duplicate values in the Date column: {duplicate_count}")


# In[20]:


# Drop any rows where Date is duplicated, keeping only the first occurrence
df_cleaned = df_cleaned.drop_duplicates(subset=['Date'], keep='first')


# In[21]:


# Display info to verify changes
print("Data types after conversion:")
print(df_cleaned.dtypes)

print("\nNumber of rows after removing duplicates based on Date:")
print(df_cleaned.shape[0])


# In[22]:


# Convert Date column into datetime64 format
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])


# In[23]:


#Convert Day_Type column into category format for memory efficiency
df_cleaned['Day_Type'] = df_cleaned['Day_Type'].astype('category')


# In[24]:


import numpy as np

# 1) Check if Date values are not in yyyy-mm-dd format
invalid_date_format = df_cleaned['Date'].isna()

# 2) Check for non-numeric values in Price, Volume, and Temperature
non_numeric_price = ~df_cleaned['Price'].apply(np.isreal)
non_numeric_volume = ~df_cleaned['Volume'].apply(np.isreal)
non_numeric_temperature = ~df_cleaned['Temperature'].apply(np.isreal)

# Combine conditions to find rows where any condition is true
invalid_rows = df_cleaned[invalid_date_format | non_numeric_price | non_numeric_volume | non_numeric_temperature]

# Display rows that meet the conditions
print("Rows with invalid date formats or non-numeric values in Price, Volume, or Temperature:")
print(invalid_rows)


# In[25]:


# Display unique values in the Day_Type column
day_type_values = df_cleaned['Day_Type'].cat.categories
print("Possible values in Day_Type:")
print(day_type_values)


# The records with "WhoAmI?" and "WhyIsDataAlwaysMessy?" are removed in the duplication removal of the Date column stage. Thus, there is no additional step required to remove them.

# # DATA ANALYSIS

# In[26]:


# Compute variation (standard deviation) for Price, Volume, and Temperature
price_variation = df_cleaned['Price'].std()
volume_variation = df_cleaned['Volume'].std()
temperature_variation = df_cleaned['Temperature'].std()

print(f"Variation in Price: {price_variation}")
print(f"Variation in Volume: {volume_variation}")
print(f"Variation in Temperature: {temperature_variation}")


# In[27]:


# Compute ratios of each Day_Type category to the sum of all records
day_type_counts = df_cleaned['Day_Type'].value_counts(normalize=True)
print("\nRatios of Day_Type values to the total number of records:")
print(day_type_counts)


# In[28]:


# Group by Day_Type and calculate the standard deviation for Price and Volume
price_volume_variation_by_day_type = df_cleaned.groupby('Day_Type', observed=False)[['Price', 'Volume']].std()

print("Variation of Price and Volume within each Day_Type category:")
print(price_volume_variation_by_day_type)


# In[29]:


import matplotlib.pyplot as plt
import os


# ## 1st Plots

# In[30]:


# Define the directory to save the plots
save_dir = '/Users/macbookpro/Desktop/MFT Energy Case Study/1st Plots'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it does not exist


# In[31]:


# Scatter plot of Price vs. Date
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Date'], df_cleaned['Price'], alpha=0.5)
plt.title("Price vs. Date")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/Price_vs_Date.png")  # Save the plot
plt.close()

# Scatter plot of Price vs. Volume
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Volume'], df_cleaned['Price'], alpha=0.5, color='orange')
plt.title("Price vs. Volume")
plt.xlabel("Volume")
plt.ylabel("Price")
plt.savefig(f"{save_dir}/Price_vs_Volume.png")  # Save the plot
plt.close()

# Scatter plot of Price vs. Temperature
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Temperature'], df_cleaned['Price'], alpha=0.5, color='green')
plt.title("Price vs. Temperature")
plt.xlabel("Temperature")
plt.ylabel("Price")
plt.savefig(f"{save_dir}/Price_vs_Temperature.png")  # Save the plot
plt.close()

# Box plot of Price vs. Day_Type
plt.figure(figsize=(8, 6))
df_cleaned.boxplot(column='Price', by='Day_Type')
plt.title("Price vs. Day_Type")
plt.suptitle("")  # Remove default boxplot title
plt.xlabel("Day_Type")
plt.ylabel("Price")
plt.savefig(f"{save_dir}/Price_vs_Day_Type.png")  # Save the plot
plt.close()


# ## Outlier Removal

# In[32]:


# Remove records with Price greater than 400
df_cleaned_no_outliers = df_cleaned[df_cleaned['Price'] <= 400]

# Further remove records with Volume greater than 1000
df_cleaned_no_outliers = df_cleaned_no_outliers[df_cleaned_no_outliers['Volume'] <= 1000]

# Display the number of records after removing outliers
print(f"Number of records after removing outliers: {df_cleaned_no_outliers.shape[0]}")

# Display the first few rows to confirm changes
print("\nFirst few rows of df_cleaned_no_outliers:")
print(df_cleaned_no_outliers.head())


# ## 2nd Plots

# In[33]:


# Define the directory to save the plots
save_dir = '/Users/macbookpro/Desktop/MFT Energy Case Study/2nd Plots'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Scatter plot of Price vs. Date
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned_no_outliers['Date'], df_cleaned_no_outliers['Price'], alpha=0.5)
plt.title("Price vs. Date")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/Price_vs_Date.png")  # Save the plot
plt.close()

# Scatter plot of Price vs. Volume
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned_no_outliers['Volume'], df_cleaned_no_outliers['Price'], alpha=0.5, color='orange')
plt.title("Price vs. Volume")
plt.xlabel("Volume")
plt.ylabel("Price")
plt.savefig(f"{save_dir}/Price_vs_Volume.png")  # Save the plot
plt.close()

# Scatter plot of Price vs. Temperature
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned_no_outliers['Temperature'], df_cleaned_no_outliers['Price'], alpha=0.5, color='green')
plt.title("Price vs. Temperature")
plt.xlabel("Temperature")
plt.ylabel("Price")
plt.savefig(f"{save_dir}/Price_vs_Temperature.png")  # Save the plot
plt.close()

# Box plot of Price vs. Day_Type
plt.figure(figsize=(8, 6))
df_cleaned_no_outliers.boxplot(column='Price', by='Day_Type')
plt.title("Price vs. Day_Type")
plt.suptitle("")  # Remove default boxplot title
plt.xlabel("Day_Type")
plt.ylabel("Price")
plt.savefig(f"{save_dir}/Price_vs_Day_Type.png")  # Save the plot
plt.close()


# ## Correlations

# In[34]:


# Compute Pearson correlation for linear relationships
pearson_corr = df_cleaned_no_outliers[['Price', 'Volume', 'Temperature']].corr(method='pearson')
print("Pearson Correlation (linear relationships):")
print(pearson_corr)

# Compute Spearman correlation for monotonic relationships
spearman_corr = df_cleaned_no_outliers[['Price', 'Volume', 'Temperature']].corr(method='spearman')
print("\nSpearman Correlation (nonlinear monotonic relationships):")
print(spearman_corr)


# # Feature Engineering

# In[35]:


# Extract Season from Date
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'


# In[36]:


# Add the Season column
df_cleaned_no_outliers['Season'] = df_cleaned_no_outliers['Date'].dt.month.apply(get_season)


# In[37]:


# Replace "Weekday" values in Day_Type with specific weekday names based on the Date column
df_cleaned_no_outliers['Day_Type'] = df_cleaned_no_outliers.apply(
    lambda row: row['Date'].day_name() if row['Day_Type'] == 'Weekday' else row['Day_Type'],
    axis=1
)

# Display the first few rows to confirm changes
print("\nFirst few rows after feature engineering:")
print(df_cleaned_no_outliers.head())


# In[38]:


# Day of the Month
df_cleaned_no_outliers['Day_of_Month'] = df_cleaned_no_outliers['Date'].dt.day


# In[39]:


# 1-Day Lagged Price
df_cleaned_no_outliers['Price_Lag_1d'] = df_cleaned_no_outliers['Price'].shift(1)


# In[40]:


# 7-Day Rolling Average of Price
df_cleaned_no_outliers['Price_7d_MA'] = df_cleaned_no_outliers['Price'].rolling(window=7, min_periods=1).mean()


# In[41]:


# Display the first few rows to verify the new features
print("\nFirst few rows after adding new features:")
print(df_cleaned_no_outliers.head(10))


# In[42]:


# Drop the first row (with NaN in Price_Lag_1d) and the Date column
data_matrix = df_cleaned_no_outliers.iloc[1:].drop(columns=['Date'])

# One-hot encode Day_Type and Season, while keeping Day_of_Month as-is
data_matrix = pd.get_dummies(data_matrix, columns=['Day_Type', 'Season'], drop_first=False)

# Display the first few rows to confirm
print("First few rows of data_matrix:")
print(data_matrix.head())


# ## Clustering

# In[43]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[44]:


# Step 1: Scale the features in data_matrix
scaler = StandardScaler()
scaled_data_matrix = scaler.fit_transform(data_matrix)

# Step 2: Determine the optimal number of clusters using Silhouette Score
silhouette_scores = []
range_clusters = range(2, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data_matrix)
    silhouette_scores.append(silhouette_score(scaled_data_matrix, kmeans.labels_))

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, silhouette_scores, marker='o', color='orange')
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()


# In[45]:


# Step 3: Choose the optimal number of clusters based on the maximum Silhouette Score
optimal_clusters = range_clusters[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters: {optimal_clusters}")


# In[46]:


# Step 4: Run K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
data_matrix['Cluster'] = kmeans.fit_predict(scaled_data_matrix)


# In[47]:


# Display the first few rows with the new Cluster feature
print("\nFirst few rows of data_matrix with Cluster feature:")
print(data_matrix.head())


# In[48]:


# Define x and y for the plot
x_feature = 'Volume'  # Change this to another feature if you have a different one in mind
y_feature = 'Price'

# Plotting
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data_matrix[x_feature], data_matrix[y_feature], c=data_matrix['Cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title(f"Clusters Visualized on {x_feature} vs. {y_feature}")
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.savefig('/Users/macbookpro/Desktop/MFT Energy Case Study/Cluster_Plot.png')
plt.show()


# ## Add the clusters as a feature to data_matrix

# In[49]:


data_matrix['Cluster'] = kmeans.labels_

# Display the first few rows to confirm the Cluster column
print("First few rows of data_matrix with Cluster column:")
print(data_matrix.head())


# ## Support Vector Regression (SVR) with a linear kernel

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[51]:


# Step 1: Define the target and features
X = data_matrix.drop(columns=['Price'])  # Features (all columns except target)
y = data_matrix['Price']                 # Target

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the SVR model with a linear kernel
svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Step 5: Make predictions and evaluate the model
y_pred_train = svr_model.predict(X_train_scaled)
y_pred_test = svr_model.predict(X_test_scaled)


# In[52]:


# Evaluation metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)


# In[53]:


print("SVR Model Evaluation (Linear Kernel):")
print(f"Training MAE: {train_mae}")
print(f"Test MAE: {test_mae}")
print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R²: {train_r2}")
print(f"Test R²: {test_r2}")


# ## Hyperparameter Tuning for the SVR with a linear kernel

# In[54]:


from sklearn.model_selection import GridSearchCV


# In[55]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the SVR model with linear kernel
svr_model = SVR(kernel='linear')

# Set up the parameter grid for C and epsilon
param_grid = {
    'C': [0.1, 1, 10, 100],          # Regularization strength
    'epsilon': [0.01, 0.1, 0.5, 1]   # Margin of tolerance
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters and the best score from GridSearch
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score (negative MAE):", grid_search.best_score_)


# In[56]:


# Train the SVR model with the best parameters
best_svr = grid_search.best_estimator_
best_svr.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_test = best_svr.predict(X_test_scaled)

# Evaluate the model
test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\nTuned SVR Model Evaluation:")
print(f"Test MAE: {test_mae}")
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")

