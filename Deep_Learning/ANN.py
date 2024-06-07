#Artificial neural network to predict if a customer would stay or leave the bank

#Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf 


# print(tf.__version__)

#Importing dataset
dataset = pd.read_csv('Deep_Learning/Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# print(X)

#Encoding Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

#One Hot encoding Geography column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


# print(X)
#Train test split of dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the ANN
#Initializing the ANN
ann_model = tf.keras.models.Sequential()


#Adding input layer and first hidden layer
ann_model.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Adding the second hidden layer
ann_model.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Adding the output layer
ann_model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))



#Training ANN

#Compiling the ANN
ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Training the ANN on training set
ann_model.fit(X_train,y_train,batch_size = 32, epochs = 100)

#Making predictions and evaluation
y_pred = ann_model.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]]))

print(f'Predicted probability - ${y_pred}')

#Predicting on the test results
y_pred = ann_model.predict(X_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print("Accuracy:",accuracy_score(y_test,y_pred))