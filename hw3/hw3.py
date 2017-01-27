import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from KNN import KNN
import sys
sys.path.append('home/Documents/Dev/School/Machine_Learning/hw3')

#
#
#
#
# Python script for training, testing and comparing the Scikit SVM and my own
# KNN alrogithms on the UCI Wine dataset.
#
#
#

#***Collect data***#
data = pd.read_csv('wine_data.csv', header=0, sep=",")

#Clean
imr = Imputer(missing_values='NaN', strategy='mean')
imr = imr.fit(data)
data = imr.transform(data.values)

#Attributes, Class and Train/Test
X = data[0:, 1:]
y = data[0:, 0]

#Split Train and Test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

#Split Test and Dev (from Test)
X_test, X_dev, y_test, y_dev = \
    train_test_split(X_test, y_test, test_size=0.5, random_state=0)

#Standardize
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_dev = sc.transform(X_dev)



#***Train SVM***#
classifier = svm.SVC(C=1, decision_function_shape="ovr")
classifier.fit(X_train, y_train)

#Test Model#
prediction = classifier.predict(X_test)

#Evaluate Errors#
correct = 0
for i, j in zip(np.nditer(prediction), np.nditer(y_test)):
    if(i == j):
        #print("{} : {}".format(i, j))
        correct = correct + 1

acc =  1.0 *correct/(y_test.size)
print("SVM accuracy: {}".format(acc))



#***KNN Model***#
knn = KNN(k=10)
predictions = knn.predict(X_train, X_test, y_train)

#Evaluate Errors#
correct = 0
for i, j in zip(np.nditer(predictions), np.nditer(y_test)):
    if(i == j):
        #print("{} : {}".format(i, j))
        correct = correct + 1

acc =  1.0 *correct/(y_test.size)
print("KNN accuracy: {}".format(acc))
