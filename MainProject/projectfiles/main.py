import csv
from logistic_regression import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
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
import pickle




if __name__ == '__main__':
    os.chdir('C:\\Users\\Enendu\\Documents\\GitHub\\sentimentanalysis\\MainProject\\projectfiles')
    filename = 'filtered_dataset1.csv'
    df = pd.read_csv(filename)

    #get the crawled tweets to clean and process it. Change to buharitweets.csv if running file for buhari
    filename2 = 'atikutweets20190108.csv'
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

    vectorizer = TfidfVectorizer(ngram_range=(1,4), analyzer='word', min_df=30)
    vectorizer.fit_transform(inputs)
    label_encoder = LabelEncoder() 
   
    
    #scaler = StandardScaler()
    #inputs = scaler.fit_transform(inputs)
    #X_test = scaler.transform(X_test)

    #mergedinput = hstack([finalcorpus, inputs])

    #finalinputs = inputs[:300000] + inputs[800001:1100001]
    #finallabels = labels[:300000] + labels[800001:1100001]

    targets = [0, 4]
   

    #Split the dataset into train and test
    input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size=0.3, shuffle=True)

    #run this line only after model has been saved 
    #label_encoder.fit_transform(label_train)

    #run this method only the first time of running this file
    def train(vectorizer, label_encoder, inputtrain, labeltrain, inputtest, labeltest, **options):        
        C_values = [1.99, 2.0, 2.05, 2.1, 2.2, 2.3, 2.78]
        train_labels = label_encoder.fit_transform(labeltrain)
        X = vectorizer.transform(inputtrain)
        X_test = vectorizer.transform(inputtest)
        model = LogisticRegressionCV(Cs=C_values, cv=5, solver='liblinear', max_iter=3000, multi_class='ovr',n_jobs=6, refit=True).fit(X, train_labels)
        #save model
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        #print prediction scores
        trainprediction = model.predict(X)
        trainresult = label_encoder.inverse_transform(trainprediction)
        print("Train Accuracy:" + " " + str(accuracy_score(labeltrain, trainresult)))
        testprediction = model.predict(X_test)
        testresult = label_encoder.inverse_transform(testprediction)
        print("Test Accuracy:" + " " + str(accuracy_score(labeltest, testresult)))
        #print contributing words
        print('Best C parameters: ' + str(model.C_))
        print(model.coef_)  
        score = model.score(X, train_labels)
        print(score)
        feature_to_coef = {
            word: coef for word, coef in zip(
                vectorizer.get_feature_names(), model.coef_[0]
            )
        }
        for best_positive in sorted(
            feature_to_coef.items(), 
            key=lambda x: x[1], 
            reverse=True)[:50]:
            print (best_positive)

        for best_negative in sorted(
            feature_to_coef.items(), 
            key=lambda x: x[1])[:20]:
            print (best_negative)

    #logistic funtion 
    #def train_logistic(inputs, input_train, input_test, label_train, label_test, targets):
    #    logreg = LogisticRegression(inputs)
    #    logreg.train(input_train, label_train)
    #    output = logreg.classify(input_test)
    #    print(logreg.count_label_occurences(labels))
    #    ##lm.parameter_tuning(training, l_training)
    #    print("Accuracy Training" + str(logreg.evaluate(input_train, label_train)))
    #    print("Accuracy Test" + str(logreg.evaluate(input_test, label_test)))
    #    print(logreg.getMetrics(input_test, label_test))
    #    print(logreg.getScore(input_test, label_test))
    #    logreg.printConfusion(input_test, label_test, targets)
    #    print (logreg.classify(['Atiku is a terrible candidate']))
     
    #function to predict tweets    
    def predict_tweets(tweetset, scorelist, mydf, vectorizer, label_encoder, query):
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        resultlist = []
        for item in tweetset:
            X = vectorizer.transform([item])
            prediction = loaded_model.predict(X)
            resultlist.append(label_encoder.inverse_transform(prediction))

        mydf['text'] = tweetset
        mydf['scorelist'] = scorelist
        mydf['sentimentprediction'] = resultlist
        mydf.to_csv(query+'results.csv', index=False, encoding='latin-1')
        
  
results = []
#take the tweets from the crawled tweets file
test = df2.Text
#perform the cleaning of the tweets
for t in test:
    results.append(tweet_cleaner_updated(t))

#change for buhari
query = 'atiku'

lastdf = pd.DataFrame() 
#I use spacy for parts of speech tagging. pip install -U spacy
nlp = spacy.load('en') 
#I use sentimental to score the sentiment of each tweet. pip install -U git+https://github.com/text-machine-lab/sentimental.git
sentiment = Sentimental(word_list = 'afinn.csv', negation = 'negations.csv')
tweetset = []
scorelist = []
for r in results:
    doc=nlp(r)
    #Part of speech taggin. get the noun subjects, noun objects and roots of the tweet
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj" or tok.dep_ == "dobj" or tok.dep_ == "ROOT") ]
    #print(r + " " + str(sub_toks))
    #check if buhari/atiku is either a subject, object or root word then filter out 
    if query in str(sub_toks):
        tweetset.append(r)
        sentence_sentiment = sentiment.analyze(r)
        scorelist.append(sentence_sentiment['score'])

#comment out train function after first run
train(vectorizer, label_encoder, input_train, label_train, input_test, label_test)
predict_tweets(tweetset, scorelist, lastdf, vectorizer, label_encoder, query)