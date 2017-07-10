from __future__ import print_function
import os
# from insightMedium_aws.medium import Medium
# from insightMedium_aws.model import Post, to_dict
from insightMedium_aws.medium import Medium

# TEXT SUMMARIZATION
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as LSA
from sumy.summarizers.lex_rank import LexRankSummarizer as LEX
from sumy.summarizers.luhn import LuhnSummarizer as LUHN
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from gensim.summarization import summarize

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
# need to import hdf?

# String fuzzy matching
import Levenshtein #pip install python-levenshtein
import difflib

# Machine learning / Model Building
from sklearn import datasets, linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
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

import SnippetIQ_Analysis_v1 as snippetanalysis

import sys
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words # pip install stop-words
from gensim import corpora, models
import gensim
import csv
import _pickle as cPickle
from sklearn.externals import joblib
from string import digits
import bz2
import pyLDAvis #pip install pyldavis
import pyLDAvis.gensim

from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

#####################################################


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def test():
    vect = TfidfVectorizer(tokenizer=LemmaTokenizer())
    print("This is a test function")

test()
