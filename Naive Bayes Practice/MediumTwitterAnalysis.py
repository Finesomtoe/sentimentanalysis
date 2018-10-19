import pandas as pd
import numpy as np
dataset = "./dataset/training.1600000.processed.noemoticon.csv"
cols = ['sentiment', 'id', 'date', 'query', 'username', 'text']
df = pd.read_csv(dataset, header=None, names=cols, encoding='latin-1')
del df['id']
del df['date']
del df['query']
del df['username']

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=1000, binary=True)
X = vect.fit_transform(df['sentiment'])
X.toarray()

from sklearn.model_selection import train_test_split
X = df['sentiment']
y = df['text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=1000, binary=True)
X_train_vect = vect.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb.score(X_train, y_train)

X_test_vect = vect.transform(X_test)
y_pred = nb.predict(X_test_vect)

print('y_pred')

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))

