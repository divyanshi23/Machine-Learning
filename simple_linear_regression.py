# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:28:13 2019

@author: hp
"""

#importing the libraries
import numpy as np   #used to include any type of mathematics in our code
import matplotlib.pyplot as plt   #used to plot different charts
import pandas as pd  #used to import and manage dataset

#importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values  #creating independent variable matrix
y=dataset.iloc[:,1].values  #creating dependent data variable matrix

#dividing the data set into train ang test sets
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=45)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training dataset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test Dataset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()