from mymodel import Model
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score  

class LogisticRegression(Model):

    def __init__(self, data):
        self.__data = data
        self.__stopwords = set(stopwords.words('english'))
        self.__bigram_vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='word')
        self.__bigrams = self.__bigram_vectorizer.fit_transform(self.__data)
        print (self.__bigrams.shape)


    def train(self, inputs, labels, **options):
        self.__label_encoder = LabelEncoder()
        self.__train_labels = self.__label_encoder.fit_transform(labels)
        X = self.__bigram_vectorizer.transform(inputs)
        self.__model = LogisticRegressionCV(cv=5, solver='sag', max_iter=200, multi_class='ovr',n_jobs=6, refit=True).fit(X, self.__train_labels)
        print('Best C parameters: ' + str(self.__model.C_))

    def classify(self, inputs):
        X = self.__bigram_vectorizer.transform(inputs)
        prediction = self.__model.predict(X)
        return self.__label_encoder.inverse_transform(prediction)

    def evaluate(self, inputs, labels):
        predicted = self.classify(inputs)
        return accuracy_score(labels, predicted)






