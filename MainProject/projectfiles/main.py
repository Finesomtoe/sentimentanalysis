import csv
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk.tokenize as nltk
import os
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    os.chdir('C:\\Users\\Enendu\\Documents\\GitHub\\sentimentanalysis\\MainProject\\projectfiles')
    filename = 'clean_tweet.csv'
    df = pd.read_csv(filename)

    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    df.info()

    inputs, labels = df.text.tolist(), df.target.tolist()

    X = []
    y = []
    for item in inputs[:400000]:
        X.append(item)
    for item in inputs[798198:1198198]:
        X.append(item)
    for item in labels[:400000]:
        y.append(item)
    for item in labels[798198:1198198]:
        y.append(item)

    targets = [0, 4]
   

    #Split the dataset into train and test
    input_train, input_test, label_train, label_test = train_test_split(X, y, test_size=.03, shuffle=True)

    logreg = LogisticRegression(inputs)
    logreg.train(input_train, label_train)
    output = logreg.classify(input_test)
   

    print(logreg.count_label_occurences(y))
    ##lm.parameter_tuning(training, l_training)
    print("Accuracy Training" + str(logreg.evaluate(input_train, label_train)))
    print("Accuracy Test" + str(logreg.evaluate(input_test, label_test)))
    print(logreg.getMetrics(input_test, label_test))
    logreg.printConfusion(input_test, label_test, targets)
    print (logreg.classify(['Atiku is a terrible candidate']))
    print (logreg.classify(['Buhari is a callous, terrible candidate']))
    print (logreg.classify(['Atiku is a good candidate']))
    print (logreg.classify(['Buhari is a wonderful candidate']))
  
