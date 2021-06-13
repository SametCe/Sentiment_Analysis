import pandas as pd
import numpy as np
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("Sentiment/yelp.csv")
df.head()


# Preprocessing
df = df[["stars", "text"]]

preprocessing = df["text"]


def firstFive(text):
    for i in range(5):
        print(text[i])
        print("------------------------")


firstFive(preprocessing)

preprocessing = df["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
preprocessing = preprocessing.str.replace("[^\w\s]"," ")
preprocessing = preprocessing.str.replace("\d","")

import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

preprocessing = preprocessing.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords.words("english")))
preprocessing = preprocessing.apply(lambda x: " ".join(WordNetLemmatizer().lemmatize(i, "v") for i in x.split()))
firstFive(preprocessing)
df["text"] = preprocessing
dff = df.copy()

# Reducing class size to 3 as negative, neutral, and positive.
dff.stars[dff.stars == 2] = 1
dff.stars[dff.stars == 4] = 5

# Creating models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

tvec = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1, 3))
cv = CountVectorizer()
lr = LogisticRegression()
mb = MultinomialNB()
rfc = RandomForestClassifier(n_estimators=40)


def s_cv(splits, X, Y, pipeline, average_method):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X, Y):
        s_fit = pipeline.fit(X[train], Y[train])
        prediction = s_fit.predict(X[test])
        scores = s_fit.score(X[test], Y[test])

        accuracy.append(scores * 100)
        precision.append(precision_score(Y[test], prediction, average=average_method) * 100)
        print('              1          2          3')
        print('precision:', precision_score(Y[test], prediction, average=None))
        recall.append(recall_score(Y[test], prediction, average=average_method) * 100)
        print('recall:   ', recall_score(Y[test], prediction, average=None))
        f1.append(f1_score(Y[test], prediction, average=average_method) * 100)
        print('f1 score: ', f1_score(Y[test], prediction, average=None))

        rms = sqrt(mean_squared_error(Y[test], prediction))
        print(rms)

    print('-' * 50)

    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))


# Vanilla pipeline with tf-idf+Logistic Regression
from sklearn.pipeline import Pipeline
original_pipeline_LR = Pipeline([
    ('vectorizer', tvec),
    ('classifier', lr)
])

s_cv(5, dff.text, dff.stars, original_pipeline_LR, 'macro')

# Oversampling pipelines with ROS and SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

ROS_pipeline_LR = make_pipeline(tvec, RandomOverSampler(random_state=777), lr)
SMOTE_pipeline_LR = make_pipeline(tvec, SMOTE(random_state=777), lr)

s_cv(5, dff.text, dff.stars, ROS_pipeline_LR, 'macro')
s_cv(5, dff.text, dff.stars, SMOTE_pipeline_LR, 'macro')

# Under-sampling pipeline with RUS
from imblearn.under_sampling import RandomUnderSampler
RUS_pipeline_LR = make_pipeline(tvec, RandomUnderSampler(random_state=777), lr)

s_cv(5, dff.text, dff.stars, RUS_pipeline_LR, 'macro')

# Learning Curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores =\
                learning_curve(estimator=SMOTE_pipeline_LR,
                               X=dff.text,
                               y=dff.stars,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.2, 1.03])
plt.tight_layout()
plt.show()

# Validation Curve
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=SMOTE_pipeline_LR,
                X=dff.text,
                y=dff.stars,
                param_name='logisticregression__C',
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1.0])
plt.tight_layout()
plt.show()

# Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     dff.text, dff.stars, test_size=0.25, random_state=42)

sum(y_train == 1), sum(y_train == 3), sum(y_train == 5)
sum(y_test == 1), sum(y_test == 3), sum(y_test == 5)

from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

model = Pipeline([
        ('vect', tfidf),
        ('sampling', SMOTE(random_state=42)),
        ('feature_selection', SelectFromModel(estimator=LogisticRegression(random_state=0, solver='liblinear'))),
        ('clf', LogisticRegression(random_state=0, solver='liblinear'))
    ])


param_grid = [{'vect__ngram_range': [(1, 3)],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 3)],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

gs_lr_tfidf = GridSearchCV(model, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))