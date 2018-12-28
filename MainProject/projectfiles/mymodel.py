from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib
import pandas as pd


class Model(object):
    
    def count_label_occurences(self, targets):
        dummies = pd.get_dummies(targets, prefix_sep='_')
        cat_dict = dict()
        for col in dummies.columns:
            count = 0
            for row in dummies[col]:
                count += row
            cat_dict[col] = count
        return cat_dict

    def getScore(self, inputs, targets):
        return accuracy_score(targets, self.classify(inputs))

    def getMetrics(self, inputs, targets):
        return classification_report(targets, self.classify(inputs))
    
    def printConfusion(self, inputs, targets,  vectorizer, labels=None):
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




