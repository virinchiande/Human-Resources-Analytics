
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\virin\\Desktop\\HR.csv")
print(data.head(5))


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


count_left = pd.value_counts(data['left'])
print(count_left)


# In[9]:


count_left.plot(kind = 'bar')
plt.show()


# In[13]:


count_number = pd.value_counts(data['number_project'])
print(count_number)
count_number.plot(kind = 'bar')
plt.show()


# In[21]:


print(data.time_spend_company.unique())
count_years = pd.value_counts(data['time_spend_company'])
print(count_years)
count_number.plot(kind = 'bar')
plt.show()


# In[27]:


count_acc = pd.value_counts(data['Work_accident'])
print(count_acc)
count_acc.plot(kind = 'bar')
plt.show()


# In[28]:


count_pro = pd.value_counts(data['promotion_last_5years'])
print(count_pro)
count_pro.plot(kind = 'bar')
plt.show()


# In[29]:


count_sal = pd.value_counts(data['salary'])
print(count_sal)
count_sal.plot(kind = 'bar')
plt.show()


# In[30]:


count_sales = pd.value_counts(data['sales'])
print(count_sales)
count_sales.plot(kind = 'bar')
plt.show()


# In[34]:


data.hist(bins =50, figsize=(20,15))
plt.show()


# In[35]:


corr = data.corr()


# In[37]:


corr["left"].sort_values(ascending = False)


# In[45]:


#we can infer from the scatter plot that ppl who spend more than 6 years are not leaving the company
data.plot(kind='scatter', x='left', y='time_spend_company', alpha = 0.4)
plt.show()


# In[46]:


data.plot(kind='scatter', x='left', y='average_montly_hours', alpha = 0.4)
plt.show()


# In[47]:


data.plot(kind = 'scatter', x='satisfaction_level', y='last_evaluation')
plt.show()


# In[48]:


data.plot(kind = 'scatter', x='satisfaction_level', y='average_montly_hours')
plt.show()

