import numpy as np
from collections import Counter

class NaiveBayes(object):

    def train(self, posDoc, negDoc):
        #Build amalgomated documents and get counts
        bigPos = ""
        nPos = 0
        for doc in posDoc:
            bigPos += doc + " "
            nPos +=1

        bigNeg = ""
        nNeg = 0
        for doc in negDoc:
            bigNeg += doc + " "
            nNeg +=1

        nDocs = nPos + nNeg
        nPos = float(nPos)
        nNeg = float(nNeg)
        nDocs = float(nDocs)

        #Prior Probabilities
        self.prior = [(nPos/nDocs), (nNeg/nDocs)]

        #Counter Objects
        bigList = bigPos.split(" ")
        posCounter = Counter(bigList)
        bigList = bigNeg.split(" ")
        negCounter = Counter(bigList)


        #Arrays for word's probabilities
        self.posWords = []
        uniPos = 0
        for word in list(posCounter.elements()):
            if word not in self.posWords:
                uniPos+=1
                self.posWords.append(word)

        self.negWords = []
        uniNeg = 0
        for word in list(negCounter.elements()):
            if word not in self.negWords:
                uniNeg+=1
                self.negWords.append(word)


        self.posWords = np.column_stack((self.posWords, np.ones(len(self.posWords))))
        self.negWords = np.column_stack((self.negWords, np.ones(len(self.negWords))))

        self.posWords = np.delete(self.posWords, 0, 0)
        self.negWords = np.delete(self.negWords, 0, 0)

        self.posDenom = float(len(bigPos.split(" "))+uniPos)
        self.negDenom = float(len(bigNeg.split(" "))+uniNeg)

        # Calculate positive word probabilities
        for word in self.posWords[0:, 0]:
            count = float(posCounter[word])
            self.posWords[self.posWords[0:, 0] == word, 1] = (count+1)/(self.posDenom)

        #Calculate negative word probabilities
        for word in self.negWords[0:, 0]:
            count = float(negCounter[word])
            self.negWords[self.negWords[0:, 0] == word, 1] = (count+1)/(self.negDenom)

    def test(self, testDoc):
        #Set Up
        classifier = [1, 0]
        classifier = np.column_stack((classifier, [1.0, 1.0]))
        output = np.ones(len(testDoc))
        iDoc= 0
        #Test each doc and classify it
        for doc in testDoc:
            for word in doc.split(" "):
                if word in self.posWords[0:, 0]:
                    temp = np.float64(self.posWords[(self.posWords[0:, 0] == word), 1])
                    classifier[0, 1] *= temp
                else:
                    classifier[0, 1] *= (1/self.posDenom)
                if word in self.negWords[0:, 0]:
                    temp = np.float64(self.negWords[(self.negWords[0:, 0] == word), 1])
                    classifier[1, 1] *= temp
                else:
                    classifier[0, 1] *= (1/self.negDenom)
            classifier[0, 1] *= self.prior[0]
            classifier[1, 1] *= self.prior[1]

            #Decide class of doc
            if (classifier[0, 1] > classifier[1,1]):
                output[iDoc] = 1
            else:
                output[iDoc] = 0

            #Reset per document variables
            iDoc +=1
            classifier[0, 1] = 1.0
            classifier[1, 1] = 1.0

        return output
