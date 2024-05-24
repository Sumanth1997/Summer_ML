print("Hello World")
#Model to predict profits of 50 startups basesd on RnD Spend, Administration spend and Marketing spend.

#Import libraries
import numpy as np
import pandas as pd
import sklearn


# import matplotlib.pyplot as plt

print(pd.__version__)
# print(sklearn.__version__)

#Import datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Training multiple linear regression model on training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

#Prediction on test set
y_pred = model.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))