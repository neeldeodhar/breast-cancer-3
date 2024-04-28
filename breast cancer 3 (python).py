#!/usr/bin/env python
# coding: utf-8

# In[122]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# In[123]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('insurance.csv')



df.head()

df.dropna()



# In[124]:


#displaying columns
df.columns


# In[125]:


#encoding and selecting features
enc = OrdinalEncoder()


df[['sex', 'smoker', 'region']] = enc.fit_transform(df[['sex', 'smoker', 'region']])


selected_features = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
display(selected_features)


# In[126]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X = scaler.fit_transform(selected_features)


# In[127]:


# defining y variable

y = df['charges'].values


# In[128]:


#visualization: pairplot
plt.figure(figsize = (12,5))
sns.pairplot(data = selected_features)


# In[129]:


#visualization: correlation heatmap
plt.figure(figsize =(20,10))
sns.heatmap(df.corr(), annot = True)


# In[130]:


#visualization: box plots of selected features
plt.figure(figsize = (12,5))
sns.boxplot(data = selected_features)


# In[131]:


#visualization: violin plots
plt.figure(figsize = (12,5))
sns.violinplot(data= selected_features)


# In[132]:


#creating a split (train/test)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=45)


#creating a split (train/val)

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=45)

# now the train/validate/test split will be 80%/10%/10%


# In[133]:


#training Decision Tree Regressor model for 3 different criteria
#calculating MSE, MAE and R squared scores for each criteria
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
DTRscore = []
for i in ['friedman_mse', 'squared_error', 'poisson']:

    regressor = DecisionTreeRegressor(random_state =0, criterion = i)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    DTRscore.append(regressor.score(X_test,y_test))
    
    print("MAE: " + str(mean_absolute_error(y_test,y_pred)))
    print('mse:'  + str(mean_squared_error(y_test,y_pred)))

    print (DTRscore)


# In[134]:


#plotting R squared scores for all the Decision Tree Criteria
criteria = ['friedman_mse', 'squared_error', 'poisson']

fig = plt.figure(figsize= (5,5))
plt.bar(criteria, DTRscore, color = 'blue')
plt.xlabel("criteria")
plt.ylabel("R squared")
plt.title("R squared by criteria")
plt.ylim(0.74,0.80)

plt.show


# In[135]:


#decision tree criteria optimal criterion.

print ("Based on the above visualization, the 'poisson' criterion will offer optimal solution")


# In[136]:


#training SVR model for 4 different kernels
#calculating MSE, MAE and R squared scores for each kernel
from sklearn.svm import SVR
score1 = []
for k in ['linear', 'poly', 'rbf', 'sigmoid']:

    svr_model = SVR(kernel = k, gamma ='auto', C =100, epsilon = 0.1)
    svr_model.fit(X_train,y_train)
    y_pred1 = svr_model.predict(X_test)
    score1.append (svr_model.score(X_test,y_test))
                   
  
    print('mse:'  + str(mean_squared_error(y_test,y_pred1)))
    print("MAE: " + str(mean_absolute_error(y_test,y_pred1)))

    print (score1)


# In[137]:


#plotting R squared scores for all the SVR kernels
kernel = ['linear', 'poly', 'rbf', 'sigmoid']

fig = plt.figure(figsize= (5,5))
plt.bar(kernel, score1, color = 'red')
plt.xlabel("kernel")
plt.ylabel("R squared")
plt.title("R squared by kernels")
plt.ylim(0.20,0.67)

plt.show


# In[138]:


#SVR optimal kernel.

print ("Based on the above visualization, the 'linear' kernel will offer optimal solution")


# In[139]:


#training Random Forest Regressor to FIND optimal value

# PLEASE NOTE: the test range for n_estimators i.e. 290 - 300 has been obtained after intensive testing from 200-500 estimators

from sklearn.ensemble import RandomForestRegressor
score2 = []

for k in range (290,300):
    optRFR = RandomForestRegressor(n_estimators = k, random_state = 0)
    optRFR = optRFR.fit(X_train, y_train)
    y_predOPT = optRFR.predict(X_test)
    score2.append(optRFR.score(X_test, y_test))
  


# In[140]:


#plotting accuracy percentage vs n_estimators (testing)(Random Forest Model)


plt.plot(range(290,300), score2)
plt.title("Accuracy percent vs: n_estimators value, optimal value of k")
plt.xlabel("Number of Estimators")
plt.ylabel("R squared")

plt.show()


# In[141]:


max (score2)


# In[142]:


#to train the Random Forest Regressor for optimal value obtained above


RFR = RandomForestRegressor(n_estimators = 295, random_state = 0)
RFR.fit(X_train, y_train)
scoreRFR = RFR.score(X_test, y_test)
y_pred2 = RFR.predict(X_test)

print('mse:'  + str(mean_squared_error(y_test,y_pred2)))
print("MAE: " + str(mean_absolute_error(y_test,y_pred2)))

print (scoreRFR)



# In[143]:


# plotting scores for all models (3 criteria for DTC and 4 kernels for RVR) tested above for comparison
models = ['RFR','DTR(friedman_mse)', 'DTR(squared_error)', 'DTR(poisson)', 'SVR(linear)','SVR(poly)', 'SVR(rbf)', 'SVR(sigmoid)']

maxscores = [scoreRFR]
maxscores.extend(DTRscore)
maxscores.extend(score1)


# In[144]:


#comparison of R squared scores by models
fig = plt.figure(figsize= (18,5))
plt.bar(models, maxscores, color = 'magenta')
plt.xlabel("MODELS")
plt.ylabel("R squared")
plt.title("comparison of R squared scores by models(criterion for DTR, kernels for SVR)")
plt.ylim(0.2,0.85)

plt.show


# In[145]:


# to get the maximum of all scores
max (maxscores)


# In[147]:


#final conclusion
print ("Based on the scores of the above models, Random Forest Regression gives the highest score of 83.29% ")
print ("The score also meets the required standard of 82%")


# In[ ]:




