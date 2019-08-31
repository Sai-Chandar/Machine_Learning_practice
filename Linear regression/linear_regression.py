import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
df = pd.read_csv("imports-85.data", header = None)
df.columns = ['symboling','Normalised-losses','make','fuel-type','aspiration','no of doors',\
            'body-style','drive-wheels','engine-location','wheel-base','length','width','height',\
            'curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke',\
            'compression-ratio','horse-power','peak-rpm','city-mpg','highway-mpg','price']
df = df[['compression-ratio','horse-power','peak-rpm','city-mpg','highway-mpg','price']]
df.replace('?', np.NaN, inplace = True)
#df.fillna(-9999, inplace = True)
df.dropna(inplace = True)
df = df.astype(float)
forecast = 'price'
X = np.array(df.drop([forecast], 1))
#print(X)
y = np.array(df[forecast])
#print(y)
#X = preprocessing.scale(X)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
#print(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
v = scaler.transform(np.array([[8.8, 95, 5423, 25, 27]], float))
#prd = clf.predict(v)
print(accuracy)
