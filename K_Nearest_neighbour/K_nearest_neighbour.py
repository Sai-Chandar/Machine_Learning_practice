import numpy as np
from sklearn import preprocessing, model_selection,neighbors
import pandas as pd

df = pd.read_csv("breast-cancer-wisconsin.data")
df.drop(["id"],1, inplace = True)
df.replace('?', np.NaN, inplace = True)
df.dropna(inplace = True)
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#prediction = clf.predict(np.array([[2,3,1,4,2,2,4,3,2],[2,3,1,3,2,1,4,2,2]]))
print(accuracy)
