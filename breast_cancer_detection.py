#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df= pd.read_csv('breastCancer.csv')


# In[3]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df['class'].value_counts()


# In[7]:


benign= df[df['class']==2]
malignant=df[df['class']==4]


# In[9]:


axes= benign.plot(kind='scatter',x='clump_thickness',y='size_uniformity',color= 'green', label='BENIGN CLASS')
malignant.plot(kind='scatter',x='clump_thickness',y='size_uniformity',color= 'red', label='MALIGNANT CLASS', ax= axes)
plt.show()


# In[10]:


df.dtypes


# In[11]:


from sklearn import preprocessing


# In[13]:


le= preprocessing.LabelEncoder()
df['bare_nucleoli']= le.fit_transform(df['bare_nucleoli'])


# In[14]:


df.dtypes


# In[15]:


df=df.drop(['id'],axis=1)


# In[16]:


df.head()


# In[21]:


x= df.drop(['class'],axis=1)
y= df['class']


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2,random_state=2)


# In[24]:


x_train.shape


# In[26]:


x_test.shape


# In[28]:


y_train.shape


# In[29]:


y_test.shape


# In[30]:


from sklearn import svm


# In[33]:


model= svm.SVC(kernel='linear',gamma='auto',C=0.3)


# 

# In[34]:


model.fit(x_train,y_train)


# In[35]:


y_predicted= model.predict(x_test)


# In[36]:


from sklearn.metrics import classification_report


# In[37]:


print(classification_report(y_test,y_predicted))


# In[ ]:




