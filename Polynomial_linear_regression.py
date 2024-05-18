print("Hello World")
#Model to predict previous salary of an interview candidate

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# print(X)
#Training linear regression model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X,y)


#Training Polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
plr_model = LinearRegression()
plr_model.fit(X_poly,y)


#Visualing linear regression
plt.scatter(X,y,color = 'red')
plt.plot(X,lr_model.predict(X),color='blue')
plt.title('Truth of bluff(Linear regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
# plt.show()


#Visualing Polynomial linear regression
plt.scatter(X,y,color = 'red')
plt.plot(X,plr_model.predict(X_poly),color='blue')
plt.title('Truth of bluff(Polynomial Linear regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
# plt.show()



#Predicting a new result with Linear regression
print(lr_model.predict([[6.5]]))


#Predicting a new result with Polynomial linear regression
print(plr_model.predict(poly_reg.fit_transform([[6.5]])))