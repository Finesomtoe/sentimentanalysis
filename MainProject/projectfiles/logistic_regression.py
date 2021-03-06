from __future__ import print_function
from mymodel import Model
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score  
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from sklearn.model_selection import learning_curve
import eli5
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.svm import LinearSVC, SVC
import pickle


class LogisticRegression(Model):

    def __init__(self, data):
        self.__data = data
        self.__stopwords = set(stopwords.words('english'))
        self.__bigram_vectorizer = TfidfVectorizer(ngram_range=(1,4), analyzer='word', min_df=10)
        self.__bigrams = self.__bigram_vectorizer.fit_transform(self.__data)
        #print(list(self.__bigram_vectorizer.get_feature_names))


    def train(self, inputs, labels, **options):
        C_values = [1.99, 2.0, 2.05, 2.1, 2.2, 2.3, 2.78]
        self.__label_encoder = LabelEncoder() 
        self.__train_labels = self.__label_encoder.fit_transform(labels)
        X = self.__bigram_vectorizer.transform(inputs)
        self.__model = LogisticRegressionCV(Cs=C_values, cv=5, solver='liblinear', max_iter=3000, multi_class='ovr',n_jobs=6, refit=True).fit(X, self.__train_labels)
        filename = 'finalized_model.sav'
        pickle.dump(self.__model, open(filename, 'wb'))
        #self.__model = LinearSVC(max_iter=2000).fit(inputs, self.__train_labels)
        #clf = grid_search.GridSearchCV(model, parameters, n_jobs=n_jobs, cv=3)
        #clf.fit(train_data, train_labels)
        #lin_svm_test = clf.score(test_data, test_labels)
        print('Best C parameters: ' + str(self.__model.C_))
        print(self.__model.coef_)  
        #score = self.__model.score(X, labels)
        #print(score)
        feature_to_coef = {
            word: coef for word, coef in zip(
                self.__bigram_vectorizer.get_feature_names(), self.__model.coef_[0]
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

        #self.plot_learning_curve(self.__model, "Learning Curves", X, self.__train_labels, cv= 5, n_jobs = 6)
               

    def classify(self, inputs):
        idx = 1
        #explainer = LimeTextExplainer(class_names=targets)
        X = self.__bigram_vectorizer.transform(inputs)
        #print(X.shape)
        prediction = self.__model.predict(X)
        #eli5.explain_weights(self.__model, vec=vectorizer, top=10)
        #print(f1_score(labels, prediction, average='micro'))
        #myarray = np.asarray(inputs)
        #myinputs = myarray.reshape(-1, 1)
        #print(type(myinputs[idx]))
        #print(self.__model.predict_proba(X[0]))
        #exp = explainer.explain_instance(str(myinputs[idx]), self.__model.predict_proba, num_features=6)
        #print('Document id: %d' % idx)
        #print('Probability(christian) =', self.__model.predict_proba([myinputs[idx]])[0,1])
        #print('True class: %s' % targets[labels[idx]])
        return self.__label_encoder.inverse_transform(prediction)
    

    def plotROC(self, input_test, label_test):
        logit_roc_auc = roc_auc_score(label_test, self.__model.predict(input_test))
        fpr, tpr, thresholds = roc_curve(label_test, self.__model.predict_proba(input_test)[:,1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        #plt.savefig('Log_ROC')
        plt.show()

    def evaluate(self, inputs, labels):
        predicted = self.classify(inputs)
        return accuracy_score(labels, predicted)
    
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)): 
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def weightmapping(self):
        xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = self.__model.predict_proba(grid)[:, 1].reshape(xx.shape)
        f, ax = plt.subplots(figsize=(8, 6))
        ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

        ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
                   cmap="RdBu", vmin=-.2, vmax=1.2,
                   edgecolor="white", linewidth=1)

        ax.set(aspect="equal",
               xlim=(-5, 5), ylim=(-5, 5),
               xlabel="$X_1$", ylabel="$X_2$")








