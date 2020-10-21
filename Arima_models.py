#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # markdown

# In[3]:


df=pd.read_csv("D:/hackathons/Train.csv")


# In[4]:


df.dtypes


# # forecast
# **steps:1** visualizing the data

# In[5]:


#outlier imputation


# In[7]:


print(df['GrocerySales'].quantile(0.10))
print(df['GrocerySales'].quantile(0.90))


# In[9]:


df["GrocerySales"] = np.where(df["GrocerySales"] <7871.0, 7871.0,df['GrocerySales'])
df["GrocerySales"] = np.where(df["GrocerySales"] >9000.0, 9000.0,df['GrocerySales'])


# In[10]:


plt.boxplot(df['GrocerySales'])


# In[16]:


df['Day']=pd.to_datetime(df['Day'])


# In[20]:


df=df.set_index(['Day'])


# In[21]:


df.plot()


# In[22]:


from statsmodels.tsa.stattools import adfuller


# In[29]:


log=np.log10(df)


# In[30]:


adfuller(df['GrocerySales'])


# In[31]:


import statsmodels.tsa.api as smt


# In[32]:


#Time Series = train_log
#d = 1

fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(log, lags=30, ax=axes[0], alpha=0.5)
smt.graphics.plot_pacf(log, lags=30, ax=axes[1], alpha=0.5, method='ols')
plt.tight_layout()


# In[33]:


import statsmodels as sm


# In[34]:


sm.tsa.statespace.api.SARIMAX


# In[39]:


df1=df[:'2001-11-16']
valid=df['2001-11-17':]


# In[79]:


model = sm.tsa.statespace.api.SARIMAX(df1, order = (2,1,3), seasonal_order= (2,0,3,31),
                                             enforce_stationarity=True,
                                             enforce_invertibility=False)


# In[80]:


results=model.fit()


# In[75]:


res=results.forecast(6)


# In[76]:


res


# In[77]:


from sklearn.metrics import mean_squared_error


# In[78]:


np.sqrt(mean_squared_error(res,valid))


# In[64]:


sales=results.forecast(96)


# In[66]:


sales.to_excel("D:/hackathons/arimamodel.xlsx")


# In[52]:


import itertools


# In[53]:


import sys

def auto_arima(timeseries, regressors=None, p=range(0, 2), d=range(0, 2), q=range(0, 2),
              P=range(0, 2), D=range(0, 1), Q=range(0, 2)):

    # Generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(P, D, Q))]
    warnings.filterwarnings("ignore") # specify to ignore warning messages

    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    best_results = None
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:

            try:
                temp_model = sm.tsa.statespace.api.SARIMAX(endog=timeseries,
                                                 exog=regressors,
                                                 order = param,
                                                 seasonal_order = param_seasonal,
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)
                temp_results = temp_model.fit()

                print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, temp_results.aic))
                
                if temp_results.aic < best_aic:
                    best_aic = temp_results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_results = temp_results
                
            except:
                #print("Unexpected error:", sys.exc_info()[0])
                continue
    print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
    print(best_results.summary().tables[0])
    print(best_results.summary().tables[1])
    return(best_results)


# In[ ]:




