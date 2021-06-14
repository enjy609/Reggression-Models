#importing the libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import seaborn as sns; sns.set_theme()


#importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['X1', 'X7'], axis=1)
X = dataset.iloc[:, :-1].values
Y = dataset.loc[:, "Y"].values

# Handling missing values in X2
# retrieve the numpy array
values = dataset['X2'].values
# define the imputer
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = KNNImputer(n_neighbors=3)
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
heat_map = sns.heatmap(x,xticklabels=x.columns,yticklabels= x.columns, annot = True)
############


x=dataset[['X4', 'X6','X11']]
y=dataset['Y']

X_train, X_test, Y_train, Y_test = train_test_split( x , y , test_size = 0.1, random_state = 0)

# Feature Scaling
norm=MinMaxScaler().fit(X_train)
X_train=norm.transform(X_train)
X_test=norm.transform(X_test)

#X_train = preprocessing.minmax_scale(X_train)
#X_test=preprocessing.minmax_scale(X_test)

poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X_train)
Xt_poly = poly_reg.fit_transform(X_test)
model = Ridge()
cv = RepeatedKFold(n_splits=10, n_repeats=3)
grid = dict()
grid['alpha'] = np.arange(0,1,0.01)
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
best_model = search.fit(X_poly,Y_train)
y_pred = search.predict(Xt_poly)
MAE = mean_absolute_error(y_pred,Y_test)
print(MAE)


plt.figure(figsize=(8,5))
plt.xlabel('X4', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.ticklabel_format(style='plain')
plt.scatter(dataset['X3'],dataset['Y'],s=15)
plt.show()
# test ben3mel transform bas mesh dot fit
# han train tany 3al data kolaha

test = pd.read_csv('test.csv')
test = test.drop(['X1', 'X7'], axis=1)
X = test.iloc[:, :-1].values

# Handling missing values in X2
# retrieve the numpy array
values = test['X2'].values
# define the imputer
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = KNNImputer(n_neighbors=3)
# transform the dataset
transformed_values = imputer.fit_transform(values.reshape(-1,1))
test['X2']=transformed_values

# Handling lf & reg values in X3
check = set()
for i in test['X3']:
    check.add(i)
test['X3']=test['X3'].replace(['LF', 'low fat'],'Low Fat')
test['X3']=test['X3'].replace(['reg'],'Regular')


# Handling missing  values in X9
#dataset['X9'].mode()
test['X9']=test['X9'].fillna(value=test['X9'].mode()[0], inplace = False)

#Label encode for X3
label_encoder = preprocessing.LabelEncoder()
test['X3'] = label_encoder.fit_transform(test['X3'])
test['X5'] = label_encoder.fit_transform(test['X5'])
test['X9'] = label_encoder.fit_transform(test['X9'])
test['X10'] = label_encoder.fit_transform(test['X10'])
test['X11'] = label_encoder.fit_transform(test['X11'])


x=test[['X4', 'X6','X11']]

# Feature Scaling

x = norm.transform(x)
x = poly_reg.transform(x)
y_pred = search.predict(x)

#############

Sample=pd.read_csv('sample_submission.csv')
Sample['Y']=y_pred
Sample.to_csv(r'sample_submission.csv', index = False)


fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(dataset['X4'],dataset['X6'],dataset['X11']) # plot the point (2,3,4) on the figure

plt.show()