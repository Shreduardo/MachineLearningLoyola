import numpy as np
from KNN import KNN
from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data[:, 1:]
y = iris.target

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3)

knn = KNN(k=5)
predictions = knn.predict(X_train, X_test, y_train)

#Evaluate Errors#
correct = 0
for i, j in zip(np.nditer(predictions), np.nditer(y_test)):
    if(i == j):
        #print("{} : {}".format(i, j))
        correct = correct + 1

acc =  1.0 *correct/(y_test.size)
print("KNN accuracy: {}".format(acc))
