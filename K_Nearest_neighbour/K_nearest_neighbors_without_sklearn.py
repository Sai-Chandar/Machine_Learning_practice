import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

dataset = {
            'r': [[1,3.5],[1,4],[1,5]],
            'y': [[3.5,1],[4,1],[5,1]]
            }


def k_nearest_neighbors(dataset, predict, k=3):
    distance = []
    for group in dataset:
        for features in dataset[group]:
            euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
##            print(euclidean_distance, group)
            distance.append([euclidean_distance, group])
    distance.sort()
    votes = [i[1] for i in distance[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


predict = [2,5]
sol = k_nearest_neighbors(dataset, predict)
print(sol)

for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0], j[1], color = i)
plt.scatter(predict[0], predict[1], color = sol, s = 20)

plt.show()
