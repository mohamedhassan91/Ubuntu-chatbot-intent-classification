import os
from pathlib import Path
import json 
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pickle
import joblib
from sklearn.pipeline import Pipeline



with open('AskUbuntuCorpus.json','r',encoding="utf8") as f:
    dataset = json.loads(f.read())
X = list(dataset.keys())
y = list(dataset.values())

def load_data():
    dataset = {dataset['sentences'][i]['text'] : dataset['sentences'][i]['intent'] for i in range(len(dataset['sentences']))}
    X = list(dataset.keys())
    y = list(dataset.values())
    return X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3))
x_train = tfidf.fit_transform(X_train)
x_test = tfidf.transform(X_test)



def train(X, y, transformers=None):
    clfs = [SGDClassifier(), LogisticRegression(), LinearSVC(),RandomForestClassifier()]
    best_f1 = 0
    best_clf = None
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    for clf in clfs:
        print(f"training clf {clf}")
        if transformers:
            clf = Pipeline(transformers+[('clf', clf)])
        clf.fit(x_train, y_train)
        f1 = f1_score(y_test, clf.predict(x_test), average='micro')
        print(f"clf {clf} has f1 score of {f1}")
        if f1> best_f1:
            best_f1 = f1
            best_clf = clf
    print(f"trained with best f1 of {best_f1}")
    return best_clf

PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickles", "model.pkl")


def save_model(clf):
    joblib.dump(clf, PICKLE_DIR, True)

def load_model():
    return joblib.load(PICKLE_DIR)

   
def run():
    # load the data
    X, y = load_data()
    # load transformers
    transformers = [
        
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2)))
    ]
    clf = train(X, y, transformers)
    save_model(clf)
    print("clf trained !!")






