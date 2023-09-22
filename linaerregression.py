# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:39:44 2023

@author: nandi
"""

#importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data from files 
data = pd.read_csv('advertising.csv')
data.head()

#visualize dataset
fig ,axis = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axis[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axis[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axis[2])

#creating x and y for linear regression
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

#importing linear regression algo
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

result = 6.77 + 0.0554*59
print(result)

#create a df with min and max value 
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds = lr.predict(X_new)
print(preds)

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,preds,c='red',linewidth = 3)

import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lm.conf_int()

# find prob values
print(lm.pvalues)

#find r-squared values
print(lm.rsquared)

#multi linear regression
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales

lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data = data).fit()
print(lm.conf_int())
print(lm.summary())

