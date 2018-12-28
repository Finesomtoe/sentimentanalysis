import csv
from logistic_regression import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from naive_bayes import NaiveBayes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk.tokenize as nltk
import os
import numpy as np
import re
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
from sentimental import Sentimental




if __name__ == '__main__':
    os.chdir('C:\\Users\\Enendu\\Documents\\GitHub\\sentimentanalysis\\MainProject\\projectfiles')
    filename = 'filtered_dataset1.csv'
    df = pd.read_csv(filename)

    #get the crawled tweets to clean and process it
    filename2 = 'crawledtweets.csv'
    df2 = pd.read_csv(filename2)
    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'https?://[^ ]+'
    combined_pat = r'|'.join((pat1, pat2))
    www_pat = r'www.[^ ]+'
    negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                    "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                    "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                    "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                    "mustn't":"must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

    #function to clean up tweets
    def tweet_cleaner_updated(text):
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
        stripped = re.sub(combined_pat, '', bom_removed)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        stop_words = set(stopwords.words('english'))
        words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
        return (" ".join(words)).strip()


    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    df.info()

    #input and label from training dataset
    foo = df.text.tolist();
    inputs, labels = df.newestcorrectedtext.tolist(), df.iloc[:,1].values

    #tfidfvectorizer = TfidfVectorizer(ngram_range=(1,4), analyzer='word', min_df=30)
    #finalcorpus = tfidfvectorizer.fit_transform(corpus)
    
    #scaler = StandardScaler()
    #inputs = scaler.fit_transform(inputs)
    #X_test = scaler.transform(X_test)

    #mergedinput = hstack([finalcorpus, inputs])

    #finalinputs = inputs[:300000] + inputs[800001:1100001]
    #finallabels = labels[:300000] + labels[800001:1100001]

    targets = [0, 4]

    
   

    #Split the dataset into train and test
    input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size=0.3, shuffle=True)

    

    #naive bayes function
    def train_naivebayes(inputs, input_train, input_test, label_train, label_test, targets):
        nb = NaiveBayes(inputs)
        nb.train(input_train, label_train)
        output = nb.classify(input_test)
        #nb.parameter_tuning(input_train, label_train)
        print("Accuracy Training" + str(nb.evaluate(input_train, label_train)))
        print("Accuracy Test" + str(nb.evaluate(input_test, label_test)))
        print(nb.getMetrics(input_test, label_test))
        nb.printConfusion(input_test, label_test, targets)

    #logistic funtion 
    def train_logistic(inputs, input_train, input_test, label_train, label_test, targets, testset, scorelist, mydf):
        logreg = LogisticRegression(inputs)
        logreg.train(input_train, label_train)
        output = logreg.classify(input_test)
        print(logreg.count_label_occurences(labels))
        ##lm.parameter_tuning(training, l_training)
        print("Accuracy Training" + str(logreg.evaluate(input_train, label_train)))
        print("Accuracy Test" + str(logreg.evaluate(input_test, label_test)))
        print(logreg.getMetrics(input_test, label_test))
        print(logreg.getScore(input_test, label_test))
        logreg.printConfusion(input_test, label_test, targets)
        print (logreg.classify(['Atiku is a terrible candidate']))
        resultlist = []
        for item in testset:
           resultlist.append(logreg.classify([item]))
        #logreg.weightmapping()
        #logreg.plotROC(input_test, label_test)
        mydf['text'] = testset
        mydf['scorelist'] = scorelist
        mydf['sentimentprediction'] = resultlist
        mydf.to_csv('testresult.csv', index=False, encoding='latin-1')
        
  
results = []
#take the tweets from the crawled tweets file
test = df2.Text
#perform the cleaning of the tweets
for t in test:
    results.append(tweet_cleaner_updated(t))

lastdf = pd.DataFrame() 
#I use spacy for parts of speech tagging. pip install -U spacy
nlp = spacy.load('en') 
#I use sentimental to score the sentiment of each tweet. pip install -U git+https://github.com/text-machine-lab/sentimental.git
sentiment = Sentimental(word_list = 'afinn.csv', negation = 'negations.csv')
newresults = []
scorelist = []
for r in results:
    doc=nlp(r)
    #Part of speech taggin. get the noun subjects, noun objects and roots of the tweet
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj" or tok.dep_ == "dobj" or tok.dep_ == "ROOT") ]
    #print(r + " " + str(sub_toks))
    #check if buhari is either a subject, object or root word then filter out 
    if 'buhari' in str(sub_toks):
        newresults.append(r)
        sentence_sentiment = sentiment.analyze(r)
        scorelist.append(sentence_sentiment['score'])