from mymodel import Model
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import numpy as np


class NaiveBayes(Model):

    #Initializing function.
    def __init__(self, data):
        self.__data = data
        self.__trigram_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
        self.__trigrams = self.__trigram_vectorizer.fit_transform(self.__data)
        print(self.__trigrams.shape)


    def train(self, inputs, labels, **options):
        self.__label_encoder = LabelEncoder()
        self.__train_labels = self.__label_encoder.fit_transform(labels)
        X = self.__trigram_vectorizer.transform(inputs)
        self.__model = MultinomialNB(alpha=1.0).fit(X, self.__train_labels)

    def parameter_tuning(self, inputs, targets, **options):
        self.__label_encoder = LabelEncoder()
        self.__train_labels = self.__label_encoder.fit_transform(targets)
        X = self.__trigram_vectorizer.transform(inputs)
        kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        evaluation = cross_val_score(self.__model, X, self.__train_labels, cv=kf)
        print(evaluation)
        param_grid = {"alpha": np.array([1, 0.1, 0.01, 0.2, 0.02, 0.3, 0.03, 0.4, 0.04, 0.5, 0.05, 0.6, 0.06, 0.7, 0.07, 0.8, 0.08, 0.9, 0.09, 0])}
        gcv = GridSearchCV(self.__model, param_grid, cv=kf)
        gcv.fit(X, self.__train_labels)
        print("Best alpha parameter" + str(gcv.best_params_))
        print("Best alpha score" + str(gcv.best_score_))

    def classify(self, inputs):
        X = self.__trigram_vectorizer.transform(inputs)
        prediction = self.__model.predict(X)
        return self.__label_encoder.inverse_transform(prediction)

    def evaluate(self, inputs, labels):
        predicted = self.classify(inputs)
        return accuracy_score(labels, predicted)






