import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection

df = pd.read_csv("../K_Nearest_neighbour/breast-cancer-wisconsin.data")
df.drop(['id'], 1, inplace = True)
df.replace('?', np.NaN, inplace = True)
df.dropna(inplace = True)
X = np.array(df.drop(['class'], 1)).astype(float)
##X = preprocessing.scale(X)
y = np.array(df['class']).astype(float)
y = np.where(y==2, 0, 1)
clf = KMeans(n_clusters = 2)
clf.fit(X)
##print(y)
correct = 0
for i in range(len(y)):
    if clf.labels_[i] == y[i]:
        correct+=1
print(correct/len(y))

        
