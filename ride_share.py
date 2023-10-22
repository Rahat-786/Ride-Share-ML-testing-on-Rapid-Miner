#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[19]:


df = pd.read_csv("Ride Share train.csv", sep=';')


# In[20]:


df.head()


# In[21]:


df.shape


# In[26]:


df.columns


# We will be using 'hour', 'destination', 'cab_type', 'distance' for analysis

# In[43]:


df['hour'].isnull().any()


# In[44]:


df['destination'].isnull().any()


# In[45]:


df['cab_type'].isnull().any()


# In[46]:


df['distance'].isnull().any()


# In[51]:


df['hour'].head(10)


# In[52]:


df['distance'].head(10)


# In[53]:


df['cab_type'].head(10)


# In[54]:


df['destination'].head(10)


# In[55]:


df['price'].head(10)


# In[96]:


df['distance'] = df['distance'].astype('float16')
print(df.dtypes)


# In[97]:


df['cab_type'] = df['cab_type'].astype('float16')
print(df.dtypes)


# In[98]:


df['price'] = df['price'].astype('float16')
print(df.dtypes)


# In[ ]:





# In[56]:


df2 = pd.read_csv("Ride Share test.csv", sep=';')


# In[57]:


df2.head()


# In[58]:


df2.columns


# In[59]:


df2['hour'].isnull().any()


# In[60]:


df2['destination'].isnull().any()


# In[61]:


df2['cab_type'].isnull().any()


# In[62]:


df2['distance'].isnull().any()


# In[63]:


df2['price'].isnull().any()


# In[65]:


mean_value = df2['price'].mean()
df2['price'].fillna(mean_value, inplace=True)


# In[67]:


df2['price'].isnull().any()  #now the columns with NaN values have been filled with the mean value.


# In[99]:


df2['distance'] = df['distance'].astype('float16')
print(df2.dtypes)


# In[100]:


df2['cab_type'] = df2['cab_type'].astype('float16')
print(df2.dtypes)


# In[101]:


df2['price'] = df2['price'].astype('float16')
print(df2.dtypes)


# In[ ]:





# In[102]:


from sklearn.preprocessing import LabelEncoder

# create an instance of the LabelEncoder
le = LabelEncoder()

# iterate through all the columns in the dataset
for col in df2.columns:
    # check if the column is categorical
    if df2[col].dtype == 'object':
        # use the LabelEncoder to transform the categorical values into numerical values
        df2[col] = le.fit_transform(df2[col])

# save the encoded dataset
df2.to_csv('my_encoded_dataset.csv', index=False)


# In[103]:


df2.head()


# In[104]:


from sklearn.preprocessing import LabelEncoder

# create an instance of the LabelEncoder
le = LabelEncoder()

# iterate through all the columns in the dataset
for col in df.columns:
    # check if the column is categorical
    if df[col].dtype == 'object':
        # use the LabelEncoder to transform the categorical values into numerical values
        df[col] = le.fit_transform(df[col])

# save the encoded dataset
df.to_csv('my_encoded_dataset2.csv', index=False)


# In[105]:


df.head()


# In[106]:


# Creating a scatter plot to see the relationship between distance and price


# In[108]:


sample = df.sample(n=1000, random_state=30)
plt.scatter(sample['distance'], sample['price'], color='red')
plt.xlabel('Distance')
plt.ylabel('Price')
plt.show()


# # Histogram chart

# In[111]:


plt.hist(df['price'])
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[ ]:





# In[137]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# In[146]:


X = df2[['cab_type','distance', 'destination']]
y = df2['price']


# In[147]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[148]:


poly = PolynomialFeatures(degree=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[149]:


model = LinearRegression()


# In[150]:


model.fit(X_train, y_train)


# In[151]:


y_pred = model.predict(X_test)


# In[152]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared:', r2)


# In[153]:


plt.scatter(X_test['distance'], y_test)
plt.plot(X_test['distance'], y_pred, color='red')
plt.xlabel('Distance')
plt.ylabel('Price')
plt.show()


# In[ ]:





# In[ ]:





# In[79]:





# In[ ]:





# In[ ]:




