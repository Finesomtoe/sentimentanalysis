import csv
from logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from naive_bayes import NaiveBayes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk.tokenize as nltk
import os
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler




if __name__ == '__main__':
    os.chdir('C:\\Users\\Enendu\\Documents\\GitHub\\sentimentanalysis\\MainProject\\projectfiles')
    filename = 'filtered_dataset1.csv'
    df = pd.read_csv(filename)

    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    df.info()

    foo = df.text.tolist();
    corpus, inputs, labels = df.newestcorrectedtext.tolist(), df.iloc[:,[7]].values, df.iloc[:,1].values
    stopwords = set(stopwords.words('english'))
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1,4), analyzer='word', min_df=30)
    finalcorpus = tfidfvectorizer.fit_transform(corpus)
    
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)
    #X_test = scaler.transform(X_test)

    mergedinput = hstack([finalcorpus, inputs])

    #finalinputs = inputs[:300000] + inputs[800001:1100001]
    #finallabels = labels[:300000] + labels[800001:1100001]

    targets = [0, 4]

    
   

    #Split the dataset into train and test
    input_train, input_test, label_train, label_test = train_test_split(mergedinput, labels, test_size=0.3, shuffle=True)

    


    def train_naivebayes(inputs, input_train, input_test, label_train, label_test, targets):
        nb = NaiveBayes(inputs)
        nb.train(input_train, label_train)
        output = nb.classify(input_test)
        #nb.parameter_tuning(input_train, label_train)
        print("Accuracy Training" + str(nb.evaluate(input_train, label_train)))
        print("Accuracy Test" + str(nb.evaluate(input_test, label_test)))
        print(nb.getMetrics(input_test, label_test))
        nb.printConfusion(input_test, label_test, targets)

    def train_logistic(vectorizer, inputs, input_train, input_test, label_train, label_test, targets):
        logreg = LogisticRegression(inputs)
        logreg.train(vectorizer, input_train, label_train)
        output = logreg.classify(input_test, vectorizer)
        print(logreg.count_label_occurences(labels))
        ##lm.parameter_tuning(training, l_training)
        print("Accuracy Training" + str(logreg.evaluate(input_train, label_train, vectorizer)))
        print("Accuracy Test" + str(logreg.evaluate(input_test, label_test, vectorizer)))
        print(logreg.getMetrics(input_test, label_test,vectorizer))
        logreg.printConfusion(input_test, label_test, vectorizer, targets)
        #logreg.weightmapping()
        #print (logreg.classify(['Atiku is a terrible candidate', -3]))
        #print (logreg.classify(['Buhari is a callous, terrible candidate', -5]))
        #print (logreg.classify(['Atiku is a good candidate', 2]))
        #print (logreg.classify(['Buhari is a wonderful candidate', 4]))
        logreg.plotROC(input_test, label_test)
        
  
    
