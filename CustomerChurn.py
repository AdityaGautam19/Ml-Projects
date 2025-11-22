#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier



# In[2]:


df = pd.read_csv(r"C:\Users\manas\Downloads\customer_churn_data.csv") 
df.head()


# In[3]:


df.info()
df.describe()
df.isnull().sum()


# In[4]:


df["InternetService"].fillna(df["InternetService"].mode()[0], inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn")
plt.title("Churn Count")
plt.show()


# In[7]:


plt.figure(figsize=(6,4))
sns.barplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()


# In[9]:


df = df.drop(columns=["CustomerID"])  

le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = le.fit_transform(df[column])


# In[10]:


X = df.drop("Churn", axis=1)   
y = df["Churn"]                #


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[12]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)


# In[15]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[16]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# In[17]:


importances = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
importances.sort_values("importance", ascending=False, inplace=True)
importances.head(10)


# In[19]:


plt.figure(figsize=(8,5))
sns.barplot(x="importance", y="feature", data=importances.head(10))
plt.title("Importantance of Features")
plt.show()


# In[21]:


sample = np.array([[34, 0, 56.2, 1085.0, 1, 1, 0, 1]]) 
sample_scaled = scaler.transform(sample)
model.predict(sample_scaled)


# In[ ]:




