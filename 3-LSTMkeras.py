
# coding: utf-8

# In[262]:


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import datetime

#Packages for pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

 # Importing the Keras libraries and packages for LSTM
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers import Dense
from keras.layers import LSTM


# In[252]:


#Read csv file
data = pd.read_csv("data.csv")


# In[253]:


#See what's inside
data


# In[254]:


Id='619882'#619882 very populated, 50070E also very populated, D25249 not much populated, 5D0EAE two points..
#set date as index
data.index
data.reset_index(inplace=True)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')


# In[255]:


subdata = data[data.RP_ENTITY_ID==Id]
date_start = '2013-01-01'
date_split   = '2016-01-01'
date_end   = '2017-02-17'
train = subdata[date_start:date_split]
test = subdata[date_split:date_end]


# In[256]:


train.fillna(0.0, inplace=True)
test.fillna(0.0, inplace=True)


# In[257]:


features = [column for column in subdata.columns]
train_features = np.delete(features, [0,1,2,32,33])
train_features


# In[543]:


X_train = train[train_features].values
Y_train = train.T1_RETURN.values
X_test = test[train_features].values
Y_test = test.T1_RETURN.values


# In[544]:


# Feature Scaling
scX = MinMaxScaler(feature_range=(0,1))
scY = MinMaxScaler(feature_range=(0,1))
X_train = np.reshape(X_train,(-1,1))
Y_train = np.reshape(Y_train,(-1,1))
X_train = scX.fit_transform(X_train)
Y_train = scY.fit_transform(Y_train)
#Reshaping Array
X_train = np.reshape(X_train, (766, 1, 29))
X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
    
##Reshaping Array
#X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))#(766, 1, 28)
#X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))#(290, 1, 28)


# In[545]:


def keras_regressor():
    regressor = Sequential()
    """
    # Adding the input layerand the LSTM layer
    regressor.add(LSTM(units = 50, activation = 'relu',
                       return_sequences=True,
                       input_shape = (X_train.shape[1],X_train.shape[2])))
    #regressor.add(Dropout(0.2))
    #regressor.add(LSTM(units = 50, return_sequences=True, activation = 'relu'))
    #regressor.add(LSTM(units = 100, activation = 'relu'))
    #regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    """
    # Adding the input layerand the LSTM layer
    regressor.add(LSTM(units = 200, activation = 'relu',
                       input_shape = (X_train.shape[1],X_train.shape[2])))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN
    #regressor.compile(loss='mae', optimizer='adam')
    regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    return regressor


# In[546]:


# Initialising the RNN
regressor = keras_regressor()
regressor.summary()


# In[547]:


# Fitting the RNN to the Training set
history = regressor.fit(X_train, Y_train, batch_size = 700, validation_data=(X_test, Y_test), epochs = 10, verbose = 1, shuffle=False)
#history = regressor.fit(X_train, Y_train, batch_size = 900, validation_data=(X_test, Y_test), epochs = 50, verbose = 1, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[548]:


from math import sqrt
from sklearn.metrics import mean_squared_error
from numpy import concatenate

# Get prediction
yhat = regressor.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
inv_yhat = scY.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
Y_test = Y_test.reshape((len(Y_test), 1))
inv_y = concatenate((Y_test, X_test[:, 1:]), axis=1)
inv_y = scY.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[510]:


#reset
X_train = train[train_features].values
Y_train = train.T1_RETURN.values
X_test = test[train_features].values
Y_test = test.T1_RETURN.values

# Getting the predicted Web View
inputs = X_test
inputs = np.reshape(inputs,(-1,1))
inputs = scX.transform(inputs)
inputs = np.reshape(inputs, (290,1,29))
y_pred = regressor.predict(inputs)
y_pred = scY.inverse_transform(y_pred)

#Visualising Result
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(Y_test, color = 'blue', label = 'Current')
ax.plot(y_pred, color = 'red', label = 'Prediction (LSTM)')
ax.set_xlabel('Date')
ax.set_ylabel('T1 return')
ax.legend()
plt.title('Prediction on test sample');
plt.show()


# In[511]:


#reset
X_train = train[train_features].values
Y_train = train.T1_RETURN.values
X_test = test[train_features].values
Y_test = test.T1_RETURN.values

# Getting the predicted Web View
inputs = X_train
inputs = np.reshape(inputs,(-1,1))
inputs = scX.transform(inputs)
inputs = np.reshape(inputs, (766,1,29))
y_pred = regressor.predict(inputs)
y_pred = scY.inverse_transform(y_pred)

#Visualising Result
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(Y_train, color = 'blue', label = 'Current')
ax.plot(y_pred, color = 'red', label = 'Prediction (LSTM)')
ax.set_xlabel('Date')
ax.set_ylabel('T1 return')
ax.legend()
plt.title('Prediction on train sample');
plt.show()


# ## Prediction files

# In[248]:


left = train.T1_RETURN.to_frame()
left.reset_index(inplace=True)
left['DATE'] = pd.to_datetime(left['DATE'])
inputs = X_train
inputs = np.reshape(inputs,(-1,1))
inputs = scX.transform(inputs)
inputs = np.reshape(inputs, (766,1,29))
right = regressor.predict(inputs)
right = pd.DataFrame(scY.inverse_transform(right))
prediction_train = pd.concat([left,right], axis=1)
prediction_train.drop_duplicates('DATE')
prediction_train.columns = ['DATE','T1_RETURN', 'T1_PREDICT']
prediction_train = prediction_train.set_index('DATE')
prediction_train.to_csv( './results/trainLSTM.csv')

left = test.T1_RETURN.to_frame()
left.reset_index(inplace=True)
left['DATE'] = pd.to_datetime(left['DATE'])
inputs = X_test
inputs = np.reshape(inputs,(-1,1))
inputs = scX.transform(inputs)
inputs = np.reshape(inputs, (290,1,29))
right = regressor.predict(inputs)
right = pd.DataFrame(scY.inverse_transform(right))
prediction_test = pd.concat([left,right], axis=1)
prediction_test.drop_duplicates('DATE')
prediction_test.columns = ['DATE','T1_RETURN', 'T1_PREDICT']
prediction_test = prediction_test.set_index('DATE')
prediction_test.to_csv( './results/testLSTM.csv')

prediction_test

