# load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# load data
df = pd.read_csv('house-data.csv')
# displaying first 5 rows of imported data
df.head(5)

# removing all the null values in the data set
df.dropna(inplace = True)
print(df.isnull().sum())

# instantiate the LabelEncoder
le = LabelEncoder()
df['CHAS'] = le.fit_transform(df['CHAS'])

# display statistical details about data 
df.describe()

# verify the data types of each variable
print(df.dtypes)

# we notice that several variables have type float - need to convert this to type integer for linear regression
columns = df.columns.values
for var in columns:
    if df[var].dtypes == 'float64':
        df[var] = pd.to_numeric(df[var], errors = 'coerce')
        df[var] = df[var].astype('int64')

# verify that all variables are converted from float64
print(df.dtypes)

# Visualization: Heatmap

sb.heatmap(df.corr(), annot = True, cmap = 'magma')
plt.savefig('heatmap.png')
plt.show()

# Visualization: Scatterplot

y = df['MEDV']
scatter_df = df.drop('MEDV', axis=1)
#sample plot for only 1 dependent variable (crime) compared with the independent y variable
sample_x = scatter_df.columns[0]
sample_plot = sb.scatterplot(sample_x, y, data=df, color='blue', s=150)
plt.title('{} / Median Sales Price'.format(sample_x), fontsize = 16)
plt.xlabel('{}'.format(sample_x), fontsize = 14)
plt.ylabel('Median Sales Price (in $1000s)', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('scatterplot.png')
plt.show()

# Visualization: Distribution plot
sb.distplot(df['MEDV'], color = 'g')
plt.title('Median Sales Price Distribution', fontsize=16)
plt.xlabel('Median Sales Price', fontsize = 14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('distplot.png')
plt.show()

#split the data into training and test set to avoid overfitting
y_var = df['MEDV'].values
x_var = df.drop('MEDV', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.3)

print(x_train.shape, y_train.shape)

#instantiate RandomForestRegressor
dt = RandomForestRegressor()

#fit random forest to training data
dt.fit(x_train, y_train)

#predict output for test data
y_predicted = dt.predict(x_test)
# determine accuracy of prediction
accuracy = dt.score(x_test, y_test)
# compute the MSE value
MSE_score = MSE(y_test, y_predicted)

print("Training Accuracy:", dt.score(x_train, y_train))
print("Testing Accuracy:", accuracy)
print("Mean Squared Error:", MSE_score.mean())