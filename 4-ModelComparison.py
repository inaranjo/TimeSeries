
# coding: utf-8

# In[99]:


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import itertools

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt


import datetime


# In[51]:


models = {
    'ARIMA',
    'LSTM',
    'LinearRegression',
    'SVR',
    'SGD',
    'KNN',
    'GBR',
    'GPR',
    'MLP'
}
#Read csv file
test={}
train={}
for model in models:
    test[("%s"%model)] = pd.read_csv("./results/test%s.csv"%model)
    train[("%s"%model)] = pd.read_csv("./results/train%s.csv"%model)
    #print(train[model].info())


# ## Mean absolute error

# In[118]:


#Fill with 0 the first entry in ARIMA prediction
train['ARIMA'].fillna(0.0, inplace=True)

xTest=[]
yTest=[]
xTrain=[]
yTrain=[]
for model in models:
    #y.append(mape(train[model].T1_PREDICT,train[model].T1_RETURN))
    y_true = test[model].T1_RETURN
    y_pred = test[model].T1_PREDICT
    print(model,mean_absolute_error(y_true, y_pred))
    yTest.append(mean_absolute_error(y_true, y_pred))
    xTest.append(model)
    y_true = train[model].T1_RETURN
    y_pred = train[model].T1_PREDICT
    print(model,mean_absolute_error(y_true, y_pred))
    yTrain.append(mean_absolute_error(y_true, y_pred))
    xTrain.append(model)

xTest,yTest = zip(*sorted(zip(xTest,yTest)))
xTrain,yTrain = zip(*sorted(zip(xTrain,yTrain)))
#print(x)
#print(y)
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(xTest,yTest, 'o',color = 'blue', label = 'Test sample')
ax.plot(xTrain,yTrain, 'o',color = 'red', label = 'Train sample')
ax.set_ylim(0,0.02)
ax.set_ylabel('mae')
ax.set_xticklabels(x,rotation=45)
ax.legend()
plt.title('Mean absolute error')
plt.grid(linestyle='dotted')
plt.show()

print(min(y))#SGD


# ## Root mean squared error

# In[119]:


xTest=[]
yTest=[]
xTrain=[]
yTrain=[]
for model in models:
    #y.append(mape(train[model].T1_PREDICT,train[model].T1_RETURN))
    y_true = test[model].T1_RETURN
    y_pred = test[model].T1_PREDICT
    print(model,sqrt(mean_squared_error(y_true, y_pred)))
    yTest.append(sqrt(mean_squared_error(y_true, y_pred)))
    xTest.append(model)
    y_true = train[model].T1_RETURN
    y_pred = train[model].T1_PREDICT
    print(model,sqrt(mean_squared_error(y_true, y_pred)))
    yTrain.append(sqrt(mean_squared_error(y_true, y_pred)))
    xTrain.append(model)

xTest,yTest = zip(*sorted(zip(xTest,yTest)))
xTrain,yTrain = zip(*sorted(zip(xTrain,yTrain)))
#print(x)
#print(y)
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(xTest,yTest, 'o',color = 'blue', label = 'Test sample')
ax.plot(xTrain,yTrain, 'o',color = 'red', label = 'Train sample')
ax.set_ylim(0,0.02)
ax.set_ylabel('msqe')
ax.set_xticklabels(x,rotation=45)
ax.legend()
plt.title('Root mean squared error')
plt.grid(linestyle='dotted')
plt.show()

print(min(y)) #SGD

