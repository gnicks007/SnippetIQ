#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Litombe
"""

from __future__ import print_function
import os

from insightMedium.medium import Medium
from insightMedium.model import Post, to_dict

# pandas / matplotlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cosine

# database
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

# storage / persist-to-disk as objects
import pickle

# String fuzzy matching
import Levenshtein #pip install python-levenshtein
import difflib

# Machine learning / Model Building
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

# For Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost

import gensim
from gensim.similarities import WmdSimilarity #wordmovers distance similarity measures
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors
from gensim import corpora, models, similarities

import itertools #for confusion matrix function
from pyemd import emd #earth-movers-distance for wordmoversdistance(WMD)

from scipy.sparse import csr_matrix

import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words # pip install stop-words
from gensim import corpora, models
import gensim
from sklearn.externals import joblib
from string import digits
import bz2
import pyLDAvis #pip install pyldavis
import pyLDAvis.gensim


#############################################################



def get_data_in_dataframe():

    with open("../data/sentenceDataframe_cleaned_resetdf.p", "rb") as input_file:
        df = pd.read_pickle(input_file)
    return df

def create_sentence_resetdf():
    with open("../data/sentenceDataframe.p", "rb") as input_file:
        sentenceDataframe2 = pd.read_pickle(input_file)


    sentDataframe_noempty = sentenceDataframe2[~(sentenceDataframe2["sentence_text"]=="")]
    resetdf = sentDataframe_noempty.reset_index()
    del resetdf["index"]

    with open("../data/sentencedataframe_resetdf.pickle", "wb") as output_file:
        pickle.dump(resetdf, output_file)

    print(resetdf.shape)
    return resetdf


def encodeTarget(df):
    # Create a label (category) encoder object: Map isHighlight = {yes, no} to numerical values {1,0}
    le = preprocessing.LabelEncoder()
    le.fit(df['isHighlight'])
    #list(le.classes_)

    #converts yes-> 1,  no->0
    y_target = le.transform(df['isHighlight'])
    return y_target


def select_feature_cols(df, cols):
    df_features = df[cols].as_matrix()
    return df_features


def select_model(model_name, Xtrain, ytrain):

    #models = ["log", "svm", "rf", "xgboost"]

    if(model_name == "log"):
        #maybe do gridsearchCV or parameter C. C = regularization
        model = LogisticRegression(penalty='l1', C=1500, class_weight="balanced")

    elif(model_name == "svm"):
        model = svm.SVC(C=10, kernel='linear', class_weight="balanced", probability=True)

    elif(model_name == "rf"):
        model = RandomForestClassifier(n_estimators=50,random_state=123)

    elif(model_name == "xgboost"):
        model = xgboost.XGBClassifier(max_depth=30, n_estimators=300, learning_rate=0.25, max_delta_step=50)

    optimize_numfeatures(model, Xtrain, ytrain)
    model.fit(Xtrain, ytrain)

    return model



def get_metrics(model, ytest, ypredict, Xtest):

    score = accuracy_score(ytest, ypredict)
    logit_roc_auc = roc_auc_score(ytest, ypredict)
    classification_report(ytest, ypredict)

    fpr, tpr, thresholds = roc_curve(ytest, model.predict_proba(Xtest)[:,1])

    # plot
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area=%0.2f)' % logit_roc_auc)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Highlights: Receiver operating characteristic')
    plt.legend(loc='lower right')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(ytest, ypredict)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ["No Highlight", "Highlight"]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def optimize_numfeatures(classfier, X, y):
    rfecv = RFECV(estimator=classfier, step=1, cv=StratifiedKFold(2), scoring='roc_auc')
    rfecv = rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print(rfecv.grid_scores_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def main():
#    feature_cols = ['post_word_count', 'read_time', 'sentiment_polarity']

    feature_cols = ["relLocationSentence", 'read_time',
       'sentence_num', 'image_count','post_word_count',
       'number_post_tags','post_number_sentences', 'sentiment_polarity','sentiment_subjectivity']

#    feature_cols = ["relLocationSentence", 'read_time',
#       'sentence_num', 'image_count','post_word_count',
#       'number_post_tags','post_number_sentences']

    df = get_data_in_dataframe()

    # add more features

    X = select_feature_cols(df, feature_cols)
    y = encodeTarget(df)

    #X_new = SelectKBest(chi2, k=2).fit_transform(X, y)

    ## Use this to automate plotting error vs. the number of training example

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2)
    model = select_model("rf", X_train, y_train)
    y_predict = model.predict(X_test)

    get_metrics(model, y_test, y_predict, X_test)



    #print(len(arr))

#    test_size = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.18]
#    score_vect = []
#    logit_roc_auc_vect = []
#
#    count = 0
#
#    for i in test_size:
#        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i,random_state=42)
#
#        model = select_model("log", X_train, y_train)
#        y_predict = model.predict(X_test)
#
#        get_metrics(model, y_test, y_predict, X_test)
#
#        #print(accuracy_score(y_test, y_predict))
#        score_vect.append(roc_auc_score(y_test, y_predict))
#        #classification_report(y_test, y_predict)
#        count+=1
#
#
#    #print(X.shape)
#    data_size = [1-x for x in test_size]
#
#    plt.plot(data_size, score_vect)

main()
