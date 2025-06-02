#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

# Load dataset from your file path
df = pd.read_csv(r"C:\Users\towhi\Downloads\sales_data_sample.csv", encoding='latin1')

# Preview the data
print("Initial shape:", df.shape)
df.head()



# In[6]:


# Convert ORDERDATE to datetime
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')

# Drop rows with null ORDERDATE or SALES
df = df.dropna(subset=['ORDERDATE', 'SALES'])

# Drop duplicates
df = df.drop_duplicates()

# Create new time-based features
df['MONTH'] = df['ORDERDATE'].dt.month
df['DAY_OF_WEEK'] = df['ORDERDATE'].dt.dayofweek

print("Cleaned shape:", df.shape)



# In[7]:


# Choose relevant features and target
features = ['QUANTITYORDERED', 'PRICEEACH', 'MONTH', 'DAY_OF_WEEK']
X = df[features]
y = df['SALES']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Train model
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Predict
lin_preds = lin_model.predict(X_test)

# Evaluate
print("🔹 Linear Regression")
print("MSE:", mean_squared_error(y_test, lin_preds))
print("R²:", r2_score(y_test, lin_preds))
print("CV Score (neg MSE):", cross_val_score(lin_model, X, y, cv=5, scoring='neg_mean_squared_error').mean())


# In[9]:


from sklearn.ensemble import RandomForestRegressor

# Train model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict
rf_preds = rf_model.predict(X_test)

# Evaluate
print("🔹 Random Forest Regressor")
print("MSE:", mean_squared_error(y_test, rf_preds))
print("R²:", r2_score(y_test, rf_preds))
print("CV Score (neg MSE):", cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error').mean())


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.barplot(x=rf_model.feature_importances_, y=features)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(8,5))
plt.scatter(y_test, rf_preds, alpha=0.6, label="Random Forest")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales - Random Forest")
plt.legend()
plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
sns.histplot(df['SALES'], kde=True, bins=30)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



# In[13]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()



# In[14]:


top_product_lines = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).head(5)

plt.figure(figsize=(8,5))
sns.barplot(x=top_product_lines.values, y=top_product_lines.index)
plt.title("Top 5 Product Lines by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Line")
plt.tight_layout()
plt.show()



# In[15]:


monthly_sales = df.groupby(df['ORDERDATE'].dt.to_period("M"))['SALES'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[16]:


plt.figure(figsize=(6,4))
sns.countplot(data=df, x='DEALSIZE', order=df['DEALSIZE'].value_counts().index)
plt.title("Deal Size Distribution")
plt.xlabel("Deal Size")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# In[17]:


# Country-wise total sales
top_countries = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Countries by Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


# In[ ]:




