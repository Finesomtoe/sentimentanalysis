import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import os

os.chdir('C:\\Users\\Enendu\\Documents\\GitHub\\sentimentanalysis\\MainProject')
dataset = "clean_tweet.csv"
df = pd.read_csv(dataset)

df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)
df.info()

X = df['text'].tolist()
y = df['target'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.2, shuffle=True)

vect = CountVectorizer()
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

nb = MultinomialNB().fit(X_train_vect, y_train)

y_pred_class = nb.predict(X_test_vect)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

print(y_test.count(0))
print(y_test.count(4))
#null_accuracy = y_test.count(0).head(1) / len(y_test)
#print('Null accuracy:', null_accuracy)

print(metrics.confusion_matrix(y_test, y_pred_class))
#X_test[y_pred_class > y_test]
#X_test[y_pred_class < y_test]

y_pred_prob = nb.predict_proba(X_test_vect)[:, 1]
print(metrics.roc_auc_score(y_test, y_pred_prob))