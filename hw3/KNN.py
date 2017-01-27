import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
import sys
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class KNN(object):

    def __init__(self, k=5):
        self.k = k


    def predict(self, train, test, classes):
        predictions = np.full(test.size, 0)
        row = 0
        #Make prediction for each test case
        for rowTest in test:
            closeness = 0
            neighbors = np.full(self.k, sys.maxint) #Neighbor distance values for comparison
            neighborClasses = np.full(self.k, 0) #Neighbor classes for prediciton

            #Find nearest neighbors in "training" data
            #Record nearest neighbor classes
            for rowTrain in train:
                closenss = euclidean(rowTrain, rowTest)
                if(closeness < np.amax(neighbors)):
                    neighbors[np.argmax(neighbors)] = closeness
                    neighborClasses[np.argmax(neighbors)] = classes[row]#Save row's class

            #Make class prediction on test, save
            predictions[row] = stats.mode(neighborClasses)[0]
            row += 1
        #Return array of predicitons
        return predictions
