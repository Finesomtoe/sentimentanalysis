from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
import pandas as pd


class Model(object):
    
    def getMetrics(self, inputs, targets):
        return classification_report(targets, self.classify(inputs))
    
    def printConfusion(self, inputs, targets, labels=None):
        if labels is None:
            conf = confusion_matrix(targets, self.classify(inputs))
        else:
            conf = confusion_matrix(targets, self.classify(inputs), labels=labels)
            print(labels)
        print(conf)
    
    def save(self, filename):
        joblib.dump(self, filename)
    
    @staticmethod
    def load(filename):
        return joblib.load(filename)




