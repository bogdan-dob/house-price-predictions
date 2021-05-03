# load the libraries
import sklearn
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston


import pandas as pd
import numpy as np

# load data
data = load_boston()

# separate the data and target values in the dataset
data, target = data.data, data.target

#split the data into training and test set to avoid overfitting
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)

print(Xtrain.shape, Ytrain.shape)

# instantiate the LabelEncoder
le = LabelEncoder()

df = pd.read_csv('house-data.csv')
y = df['MEDV']
df = df.drop(['MEDV'], axis=1)
df['CHAS'] = le.fit_transform(df['CHAS'])

x = df

#instantiate RandomForestRegressor
dt = RandomForestRegressor()

#fit random forest to training data
dt.fit(Xtrain, Ytrain)

#predict output for test data
y_predicted = dt.predict(Xtest)
# determine accuracy of prediction
accuracy = dt.score(Xtest, Ytest)
# compute the MSE value
MSE_score = MSE(Ytest, y_predicted)

print("Training Accuracy:", dt.score(Xtrain, Ytrain))
print("Testing Accuracy:", accuracy)
print("Mean Squared Error:", MSE_score.mean())