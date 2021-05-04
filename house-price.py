# load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
from sklearn.metrics import explained_variance_score as evs 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge

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

# summary of data
df.info()

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
x_columns = scatter_df.columns

for i in x_columns:
    sample_plot = sb.scatterplot(i, y, data=df, color='blue', s=150)
    plt.title('{} / Median Sales Price'.format(i), fontsize = 16)
    plt.xlabel('{}'.format(i), fontsize = 14)
    plt.ylabel('Median Sales Price (in $1000s)', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig('scatterplot-{}.png'.format(i))
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

print("Skewness:", df['MEDV'].skew())
print("Kurtosis:", df['MEDV'].kurt())

#split the data into training and test set to avoid overfitting
y_var = df['MEDV'].values
x_var = df.drop('MEDV', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.5, random_state=0)

# Modelling: Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_yhat = rf.predict(x_test)

# Modelling: Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=8)
dt.fit(x_train, y_train)
dt_yhat = dt.predict(x_test)

# Modelling: Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_yhat = lr.predict(x_test)

# Modelling: Bayesian
bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
bayesian_yhat = bayesian.predict(x_train)

# Determining accuracy

rf_accuracy = rf.score(x_test, y_test)
rf_evs = evs(y_test, rf_yhat)

print("Random Forest Training Accuracy:", rf.score(x_train, y_train))
print("Random Forest Testing Accuracy:", rf_accuracy)
print("Random Forest Explained Variance Score:", rf_evs)

dt_accuracy = dt.score(x_test, y_test)
dt_evs = evs(y_test, dt_yhat)

print("Decision Tree Training Accuracy:", dt.score(x_train, y_train))
print("Decision Tree Testing Accuracy:", dt_accuracy)
print("Decision Tree Explained Variance Score:", dt_evs)

lr_accuracy = lr.score(x_test, y_test)
lr_evs = evs(y_test, lr_yhat)

print("Linear Regression Training Accuracy:", lr.score(x_train, y_train))
print("Linear Regression Testing Accuracy:", lr_accuracy)
print("Linear Regression Explained Variance Score:", lr_evs)

bayesian_accuracy = bayesian.score(x_test, y_test)
bayesian_evs = evs(y_test, bayesian_yhat)

print("Bayesian Training Accuracy:", bayesian.score(x_train, y_train))
print("Bayesian Testing Accuracy:", bayesian_accuracy)
print("Bayesian Explained Variance Score:", bayesian_evs)

