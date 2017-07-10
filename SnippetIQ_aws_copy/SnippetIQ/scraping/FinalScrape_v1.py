#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:23:14 2017

@author: Litombe
"""

#from __future__ import print_function
from insightMedium.medium import Medium
from insightMedium.model import Post, to_dict
from selenium import webdriver

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser

from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.lsa import LsaSummarizer as LSA
from sumy.summarizers.lex_rank import LexRankSummarizer as LEX
from sumy.summarizers.luhn import LuhnSummarizer as LUHN

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from nltk.corpus import words
from nltk.corpus import stopwords

import random
import pickle

import h5py
import time

mymediumA = Medium()
exec_path = "/Applications/chromedriver"
browserdriverA = webdriver.Chrome(exec_path)

def loadFileIntoPandas(filename):
    dataframe = pd.read_csv(filename, sep=" ", header=None)

    return dataframe


#fileref = open('postsSearch3.txt', 'w', encoding='utf-8')
df = loadFileIntoPandas('postsSearchTags1 copy.txt')
df.rename(columns={0: 'post_id', 1: 'NaNs', 2: 'Urls'}, inplace=True)
dfnew = df[['post_id','Urls']]
df_no_duplicates = dfnew.drop_duplicates('post_id')
df_sorted_noduplicates = df_no_duplicates.sort_values('post_id')


postUrls = df_sorted_noduplicates
postUrls = postUrls.reset_index()

del postUrls["index"]


postings = []
counter=0
numParsed = 0

for row_id, row in enumerate(postUrls.values):
    url = row.item(1)
    counter+=1
    print("Rows checked: ", counter)
    try:
        obj = mymediumA.get_parsed_single_post(url, browserdriverA)
        postings.append(obj)
        numParsed+=1
        print("Num of parsed urls: ", numParsed)

        #save every 50 rows
        if(numParsed%50 == 0):
            pickle.dump(postings, open( "finalscrapePostings.p", "wb" ))

    except Exception as e:
        print("Problem with: ", counter)
        print("Exception: ", e)
        print(type(e))
        #print("getMessage method: ", e.message, e.getMessage())
       #if(browserdriverA == None):
        print("chromedriver shutting off")
        browserdriverA.quit()
        browserdriverA = webdriver.Chrome(exec_path)
        print("chrome webdriver restarted")
        time.sleep(8) #delay in seconds
        continue
