#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:02:19 2017

@author: Litombe
"""
from __future__ import print_function
import SnippetIQ_Analysis_v1 as snippetanalysis

import os

from insightMedium.medium import Medium
from insightMedium.model import Post, to_dict

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
from sklearn.model_selection import train_test_split as ttsplit #to distinguish
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

import gensim
from gensim.similarities import WmdSimilarity #wordmovers distance similarity measures
from gensim.models.word2vec import Word2Vec
from pyemd import emd #earth-movers-distance for wordmoversdistance(WMD)


mykeys = ["post_id", "title", "post_date", "url",
          "recommend_count", "response_count", "read_time",
          "word_count", "image_count", "detectedLanguage",
          "post_content", "post_tags",
          "post_creatorId", "post_username",
          "blockquote_list", "tophighlight", "existTophighlight"]


#highly shared post is 260 rows
def toSentenceDataFrame(dataframe):
    size = len(dataframe)
    # word_count will be renamed to post_word_count to specify it's for len(post_content)
    colsOfnewDf = ['post_date', 'post_id', 'post_num', 'isHighlight','relLocationSentence', 
                   'relLengthSentence','post_content', 'read_time', 'sentence_num',
                   'sentence_text', 'image_count', 'subtitle', 'title', 'tophighlight', 
                   'post_word_count', 'post_tags',
                   'response_count', 'number_post_tags', 'url', 'recommend_count',
                   'post_number_sentences','relLocationHighlight', 'relLengthHighlight']

    df = pd.DataFrame(columns= colsOfnewDf)

    counter = 0 #counting the number of posts incorporated into new dataframe
    totSentences = 0
    threshold = 0.8
    totHighlights = 0 #just counting the total number of highlight sentences split
    newindex = 0

    for idx in range(0, size):
        rowitem = dataframe.loc[idx, "post_content"]
        highlight = dataframe.loc[idx, "tophighlight"]

        highlightArray = highlight.split(".")
        rowcontentArray = rowitem.split(".")

        messageLength = len(rowcontentArray)
        numHighlights = len(highlightArray) #this is the number of sentences associated with each post highlight

        totSentences+=messageLength
        totHighlights+=numHighlights

        for c, content in enumerate(rowcontentArray):
           
            newindex+=1
            senlength = len(content)
            
            df.loc[newindex, "post_date"] = dataframe.loc[idx, "post_date"]
            df.loc[newindex, "post_id"] = dataframe.loc[idx, "post_id"]
            df.loc[newindex, "post_num"] = idx
            df.loc[newindex, "sentence_num"] = c
            df.loc[newindex, "sentence_text"] = content
            df.loc[newindex, "isHighlight"] = "no"
            df.loc[newindex, "read_time"] = dataframe.loc[idx, "read_time"]
            df.loc[newindex, "relLocationHighlight"] = dataframe.loc[idx, "relLocationHighlight"]
            df.loc[newindex, "relLengthHighlight"] = dataframe.loc[idx, "relLengthHighlight"]
            df.loc[newindex, "relLocationSentence"] = (c+1)/messageLength
            df.loc[newindex, "relLengthSentence"] = senlength
            df.loc[newindex, "post_word_count"] = dataframe.loc[idx, "word_count"]
            df.loc[newindex, "image_count"] = dataframe.loc[idx, "image_count"]
            df.loc[newindex, "subtitle"] = dataframe.loc[idx, "subtitle"]
            df.loc[newindex, "title"] = dataframe.loc[idx, "title"]
            df.loc[newindex, "post_tags"] = dataframe.loc[idx, "post_tags"]
            df.loc[newindex, "response_count"] = dataframe.loc[idx, "response_count"]
            df.loc[newindex, "recommend_count"] = dataframe.loc[idx, "recommend_count"]
            df.loc[newindex, "number_post_tags"] = dataframe.loc[idx, "number_post_tags"]
            df.loc[newindex, "url"] = dataframe.loc[idx, "url"]
            df.loc[newindex, "post_number_sentences"] = dataframe.loc[idx, "number_sentences"]
            df.loc[newindex, "post_content"] = dataframe.loc[idx, "post_content"]
            df.loc[newindex, "tophighlight"] = dataframe.loc[idx, "tophighlight"]

            for h, highlite in enumerate(highlightArray):
                print(newindex)
                ratio = snippetanalysis.stringMatchOverlap(content, highlite)

                nontrivial = len(highlite)>1 # make sure you're not matching letters
                nonempty = highlite is not ""
                containedin = highlite in content # need this in case only a subset of the sentence is highlighted
                abovethreshold = (ratio > threshold) or containedin

                if(abovethreshold and nonempty and nontrivial):
                    #print("Match: ", ratio, "\t", idx, "\t", c, "\t", h)  
                    df.loc[newindex, "isHighlight"] = "yes"
                    counter+=1
                    
    print("total sentences: ", newindex, "Total matched highlights: ", counter, "\n")
    print("total highlights: ", totHighlights, "\n") 
    
    
    
    return df



def main():
    with open("df_highlyshared_june18.p", "rb") as f:
        postDataframe = pickle.load(f)
    
    sentenceDataframe = toSentenceDataFrame(postDataframe)
    pickle.dump(sentenceDataframe,open("sentenceDataframe.p", "wb" ))

main()

































