from NaiveBayes import NaiveBayes
import csv
import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
sys.path.append("home/Documents/Dev/School/Machine_Learning/hw4")

#Open Data
with open("PosNoStop.txt", "r") as f:
    pos = f.read()
    f.close()

with open("NegNoStop.txt", "r") as f:
    neg = f.read()
    f.close()

#Create Data structures
pos = np.array(pos.split("\n"))
neg = np.array(neg.split("\n"))

posTrain, posTest = \
    train_test_split(pos, test_size=0.3)

negTrain, negTest = \
    train_test_split(neg, test_size=0.3)

comparitor = np.concatenate((np.ones(posTest.size), np.zeros(negTest.size)))
test = np.concatenate((posTest, negTest))

nb = NaiveBayes()
nb.train(posTrain, negTrain)

output = nb.test(posTrain)

count = 0
for i in range(comparitor.size):
    if (output[i] == comparitor[i]):
        count += 1

print(float(count)/float(comparitor.size))
