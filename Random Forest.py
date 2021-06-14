#importing the libraries
import numpy as np
import pandas as pd
from numpy import nan
from sklearn.model_selection import  GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.tree import  DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor




#importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['X1', 'X7'], axis=1)
X = dataset.iloc[:, :-1].values
Y = dataset.loc[:, "Y"].values

# Handling missing values in X2
# retrieve the numpy array
values = dataset['X2'].values
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='mean')
# transform the dataset
transformed_values = imputer.fit_transform(values.reshape(-1,1))
dataset['X2']=transformed_values

# Handling lf & reg values in X3
check = set()
for i in dataset['X3']:
    check.add(i)
dataset['X3']=dataset['X3'].replace(['LF', 'low fat'],'Low Fat')
dataset['X3']=dataset['X3'].replace(['reg'],'Regular')


# Handling missing  values in X9
#dataset['X9'].mode()
dataset['X9']=dataset['X9'].fillna(value=dataset['X9'].mode()[0], inplace = False)


#Label encode for X3
label_encoder = preprocessing.LabelEncoder()
dataset['X3'] = label_encoder.fit_transform(dataset['X3'])
dataset['X5'] = label_encoder.fit_transform(dataset['X5'])
dataset['X9'] = label_encoder.fit_transform(dataset['X9'])
dataset['X10'] = label_encoder.fit_transform(dataset['X10'])
dataset['X11'] = label_encoder.fit_transform(dataset['X11'])

#Heatmap
x=dataset.corr().abs()

x=dataset[['X4', 'X6','X11']]
y=dataset['Y']


X_train, X_test, Y_train, Y_test = train_test_split( x , y , test_size = 0.1, random_state = 0)


regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# fit the regressor with x and y data
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
MSE = mean_absolute_error(Y_test,y_pred)
print(MSE)


