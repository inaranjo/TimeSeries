
# coding: utf-8

# In[52]:


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import datetime

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor

get_ipython().magic(u'matplotlib inline')


# In[2]:


#Read csv file
data = pd.read_csv("data.csv")


# In[3]:


#See what's inside
data


# In[223]:


Id='619882'#619882 very populated, 50070E also very populated, D25249 not much populated, 5D0EAE two points..
#set date as index
data.index
data.reset_index(inplace=True)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')


# In[224]:


subdata = data[data.RP_ENTITY_ID==Id]
date_start = '2013-01-01'
date_split   = '2016-01-01'
date_end   = '2017-02-17'
train = subdata[date_start:date_split]
test = subdata[date_split:date_end]


# In[225]:


train


# In[227]:


train.fillna(0.0, inplace=True)
test.fillna(0.0, inplace=True)


# In[244]:


features = [column for column in subdata.columns]
train_features = np.delete(features, [0,1,2,32,33])
train_features.shape


# In[229]:


X_train = train[train_features].values
Y_train = train.T1_RETURN.values
X_test = test[train_features].values
Y_test = test.T1_RETURN.values


# In[249]:


regressors = {
    'LinearRegression' : LinearRegression(normalize=True),
    'SVR' : svm.SVR(kernel='rbf',C=1000,gamma=.1),
    'SGD' : SGDRegressor(alpha=0.0001, average=False, epsilon=0.1, eta0=0.01,
                         loss="squared_loss", penalty=None, shuffle=False, tol=None),
    'KNN' : KNeighborsRegressor(n_neighbors=5),
    'GBR' : GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0, loss='ls'),
    'GPR' : GaussianProcessRegressor(kernel= C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                              n_restarts_optimizer=9),
    'MLP' : MLPRegressor(hidden_layer_sizes=(100, 200, 100), activation='relu', 
                         solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)
}

def mape(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return 100*np.mean(np.abs(ypred[idx]-ytrue[idx])/ytrue[idx])

for i,c in regressors.items():
    print 'fitting -- ', i , 
    c.fit(X_train, Y_train)
    print ' : error is %0.2f%%' % mape(c.predict(X_train),Y_train)


# In[248]:


for i,c in regressors.items(): 
    fig, ax = plt.subplots(figsize=(15,3))
    #f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, figsize=(15,6))
    ax.set_title('model : %s' % i)
    ax.plot(Y_test, label='Current', color='blue')
    ax.plot(c.predict(X_test), '-', color='red', label='Prediction (%s)'%i)
    ax.legend(loc='upper right')
    ax.set_ylabel('T1 return')
    ax.set_ylim([Y_test.min()*1.2,Y_test.max()*1.2])
    #ax2.plot((Y_test[-1000:] - c.predict(X_test[-1000:])) , label ='')
    #ax2.set_ylabel('residuals')
    #ax2.set_ylim([-0.02,0.02])
    #f.subplots_adjust(hspace=0)
    plt.title('Prediction on test sample');
    plt.show()


# In[243]:


for i,c in regressors.items(): 
    fig, ax = plt.subplots(figsize=(15,3))
    ax.set_title('model : %s' % i)
    ax.plot(Y_train, label='Current', color='blue')
    ax.plot(c.predict(X_train), '-', color='red', label='Prediction (%s)'%i)
    ax.legend(loc='upper right')
    ax.set_ylabel('T1 return')
    ax.set_ylim([Y_train.min()*1.2,Y_train.max()*1.2])
    #ax2.plot((Y_train[-1000:] - c.predict(X_train[-1000:])) , label ='')
    #ax2.set_ylabel('residuals')
    #ax2.set_ylim([-0.02,0.02])
    #f.subplots_adjust(hspace=0)
    plt.title('Prediction on train sample');
    plt.show()


# ## Prediction files

# In[239]:


for i,c in regressors.items(): 
    
    left = train.T1_RETURN.to_frame()
    left.reset_index(inplace=True)
    left['DATE'] = pd.to_datetime(left['DATE'])
    right = pd.DataFrame(c.predict(X_train))
    prediction_train = pd.concat([left,right], axis=1)
    prediction_train.drop_duplicates('DATE')
    prediction_train.columns = ['DATE','T1_RETURN', 'T1_PREDICT']
    prediction_train = prediction_train.set_index('DATE')
    prediction_train.to_csv( './results/train%s.csv'%i)
    
    left = test.T1_RETURN.to_frame()
    left.reset_index(inplace=True)
    left['DATE'] = pd.to_datetime(left['DATE'])
    right = pd.DataFrame(c.predict(X_test))
    prediction_test = pd.concat([left,right], axis=1)
    prediction_test.columns = ['DATE','T1_RETURN', 'T1_PREDICT']
    prediction_test = prediction_test.set_index('DATE')
    prediction_test.to_csv( './results/test%s.csv'%i)   

prediction_test

