print("Hello World")

#Model to predict previous salary of an interview candidate
#Dataset = Position_Salaries.csv

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# print(X)
# print(y)
#Feature scaling
y = y.reshape(len(y),1)
# print(y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# print(X)
# print(y)

#Training the SVR model on the whole dataset
from sklearn.svm import SVR
model = SVR(kernel='rbf') #rbf = Gaussian Radial Basis Function
model.fit(X,y)

#Predicting
y_pred = sc_y.inverse_transform(model.predict(sc_X.transform([[6.5]])).reshape(-1,1))
# print(y_pred)


#Visualisation of SVR results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color = 'red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(model.predict(X).reshape(-1,1)),color='blue')
plt.title('Truth of bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


