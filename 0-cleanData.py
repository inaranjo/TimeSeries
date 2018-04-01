
# coding: utf-8

# # Retrieve and clean data

# This code retrieves "SampleDataSet.csv", clean the data and plot some interesting observables.

# In[151]:


import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
import datetime

from statsmodels.tsa.stattools import adfuller
from pandas.core import datetools

get_ipython().magic(u'matplotlib inline')


# In[174]:


#Read csv file
data = pd.read_csv("SampleDataSet.csv")


# In[175]:


#See what's inside
data


# We have a file with 1.7M rows and 40 columns

# In[154]:


#Check the time periods
print(min(data.DATE))
print(max(data.DATE))


# Time period extends from 2005 to 2017

# In[155]:


data.info()


# In[156]:


#Check how many entities we are looking at
len(set(data.RP_ENTITY_ID))


# In[157]:


#Check the missing values, only entity and global all feature are complete
data.isnull().sum()


# 8 features are always empty, we do not consider them for our analysis as they do not add extra information.

# In[158]:


#drop columns with all NaN values
data = data.drop(labels=['GROUP_AM_ALL',
                         'GROUP_AM_HEAD',
                         'GROUP_AM_ALL_SG90',
                         'GROUP_AM_HEAD_SG90',
                         'GROUP_AM_BODY_SG90',
                         'GROUP_AM_ALL_SG365',
                         'GROUP_AM_HEAD_SG365',
                         'GROUP_AM_BODY_SG365'], axis=1)


# In[159]:


# group rows for a given asset
data=data.groupby(['DATE','RP_ENTITY_ID'], as_index=False).first()


# We group our data in order to have only one entry per asset Id and date

# In[172]:


#Check output: we have now an entry per date, per asset, filling the features with the first non null value
data


# In[161]:


#lets see if there are any more columns with missing values 
null_columns=data.columns[data.isnull().any()]
data.isnull().sum()


# In[162]:


#Check how many entries we have for each Id after row merging
from collections import Counter
print(Counter(data.RP_ENTITY_ID))


# In[163]:


def test_stationarity(timeseries):
    
    rol=28
    #Determing rolling statistics with 1 month window: 28 working days
    rolmean = timeseries.rolling(window=rol,center=False).mean()
    rolstd = timeseries.rolling(window=rol,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


# We chose to work with the more populated asset Id. First let's have a look at the target value and test its stationarity over time.

# In[164]:


Id='619882'#619882 very populated, 50070E also very populated, D25249 not much populated, 5D0EAE two points..
#set date as index
data.index
data.reset_index(inplace=True)
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')
data_test = data.replace(np.nan, 0.0)
test_stationarity(data_test[data_test.RP_ENTITY_ID==Id].T1_RETURN)


# In[165]:


#zoom
date_start = '2013-01-01'
date_end   = '2017-02-17'
data_zoom=data[date_start:date_end]


# In[166]:


data_zoom = data_zoom.replace(np.nan, 0.0)
test_stationarity(data_zoom[data_zoom.RP_ENTITY_ID==Id].T1_RETURN)


# Let's have a look at the features and their correlations.

# In[167]:


corr=data[data.RP_ENTITY_ID==Id ].corr()
plt.figure(figsize=(30, 30))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[169]:


subdata = data[data.RP_ENTITY_ID==Id]


# In[170]:


#Correlation plots for target with all features
features = [column for column in subdata.columns]
fy = 'T1_RETURN'    
for j,fx in enumerate(features) :
        if fx == 'RP_ENTITY_ID': continue
        plt.figure(figsize=(6,6))
        plt.plot(data[fx], data[fy], '.', alpha = 0.5)
        plt.xlim([subdata[fx].min(), subdata[fx].max()])
        plt.ylim([subdata[fy].min(), subdata[fy].max()])
        plt.xlabel(fx)
        plt.ylabel(fy)
        plt.show()


# In[171]:


#Correlation plots for all features
features = [column for column in subdata.columns]
for i,fx in enumerate(features) :
    for j,fy in enumerate(features) :
        if fx == 'RP_ENTITY_ID': continue
        if fx == fy or i > j : continue
        plt.figure(figsize=(6,6))
        plt.plot(data[fx], data[fy], '.', alpha = 0.5)
        plt.xlim([subdata[fx].min(), subdata[fx].max()])
        plt.ylim([subdata[fy].min(), subdata[fy].max()])
        plt.xlabel(fx)
        plt.ylabel(fy)
        plt.show()


# Let's have a look at the number of assets versus time:

# In[62]:


#Number of assets vs time
def AssetQuantity(time):
    n=0
    time = pd.to_datetime(time)    
    for asset in data.RP_ENTITY_ID.unique():
        if time < max(data[data.RP_ENTITY_ID==asset].index) and         time > min(data[data.RP_ENTITY_ID==asset].index) :
                n=n+1
    return n


# In[ ]:


#Example
AssetQuantity('2005-01-11')


# In[ ]:


#Number of assets vs time
x = [min(data.index) + datetime.timedelta(hours=24*365*i) for i in range(13)]
y = [AssetQuantity(x[i]) for i,_ in enumerate(x)]

# plot
plt.plot(x,y)
# beautify the x-labels
#plt.gcf().autofmt_xdate()

plt.show()


# Let's check the distribution of the first and last entry for each asset

# In[88]:


df = pd.DataFrame(columns=['asset','t0','t1','deltat'])
i=0
for asset in data.RP_ENTITY_ID.unique():
    tmin=min(data[data.RP_ENTITY_ID==asset].index)
    tmax=max(data[data.RP_ENTITY_ID==asset].index)
    print(asset,(tmax-tmin).days)
    df.loc[i] = [asset,tmin,tmax,tmax-tmin]
    i=i+1


# In[148]:


fig, ax = plt.subplots(figsize=(15,3))
df['t0'].hist(bins=150,label='t0',alpha = 0.5, bottom=0.1)
df['t1'].hist(bins=150,label='t1',alpha = 0.5, bottom=0.1)
ax.legend(loc='best')
#plt.ylim(0,50)
#plt.xlim('2008','2012')
#plt.yscale('log')
ax.set_yscale('log')
plt.show()


# In[150]:


fig, ax = plt.subplots(figsize=(15,3))
df['deltat'].astype('timedelta64').hist(bins=50,label='lifetime',alpha = 0.5, bottom=0.1)
ax.legend(loc='best')
#ax.set_yscale('log')
plt.show()


# Save data:

# In[ ]:


data.to_csv( './data.csv')

