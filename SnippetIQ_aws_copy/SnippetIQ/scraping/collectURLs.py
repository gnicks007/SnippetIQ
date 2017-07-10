#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 08:15:27 2017

@author: Litombe
"""

from __future__ import print_function
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

fileref = open('postsSearchTags1.txt', 'w', encoding='utf-8')
tagnumresultref = open('tag_numresults.txt', 'w', encoding='utf-8')
searchwordsref = open('searchwordsused.txt', 'w', encoding='utf-8')
allpostsref = open('allpostspickle', 'w', encoding='utf-8')

mymediumB = Medium()
exec_path = "/Applications/chromedriver"
browserdriverB = webdriver.Chrome(exec_path)


mykeys = ["post_id", "title", "post_date", "url",
          "recommend_count", "response_count", "read_time",
          "word_count", "image_count", "detectedLanguage",
          "post_content", "post_tags",
          "post_creatorId", "post_username",
          "blockquote_list", "tophighlight", "existTophighlight"]

def loadFileIntoPandas(filename):
    dataframe = pd.read_csv(filename, sep=" ", header=None)

    return dataframe


def populateDataframe(postObjectList):
    colNames = ['post_date', 'post_creatorId', 'post_id', 'post_content',
            'post_username', 'read_time', 'detectedLanguage',
            'blockquote_list', 'existTophighlight', 'image_count',
            'subtitle', 'title', 'number_blockquotes', 'tophighlight',
            'word_count', 'post_tags', 'response_count', 'number_post_tags',
            'url', 'recommend_count']
    postdf = pd.DataFrame(columns=colNames)

    objColNames = ['blockquote_list', 'post_tags']
    numericColNames = ['post_date','read_time', 'image_count',
                       'number_blockquotes','word_count',
                       'response_count', 'number_post_tags','recommend_count']

    postdf[objColNames] = postdf[objColNames].astype(object)
    postdf[numericColNames] = postdf[numericColNames].astype(float)

    for post in postObjectList:
        try:
            post_dict = to_dict(post)
            #print(post_dict.keys())
            removeImage = post_dict.pop("preview_image", None)
            numTags = len(post_dict["post_tags"])
            numBlockquotes = len(post_dict["blockquote_list"])

            post_dict["number_post_tags"] = numTags
            post_dict["number_blockquotes"] = numBlockquotes

            # Have values in case you need to check for missing values

            if(numBlockquotes==0):
                post_dict["blockquote_list"] = np.NAN

            if(numTags==0):
                post_dict["post_tags"] = np.NAN

            if(not post_dict["tophighlight"]):
                post_dict["tophighlight"] = np.NAN

            postdf = postdf.append(post_dict, ignore_index=True)
            #print(newpostdf)
            #quotes = newpostdf["blockquote_list"][0]
            #firstquote = quotes[0]
        except:
            continue

    return postdf

def getPostsByTags(keywordList, count, allPosts):

    counter = 0
    for word in keywordList:
        try:
            posts = mymediumB.get_posts_by_tag(word, count)
            if(posts is not None):
                allPosts.append(posts)


                pickle.dump(allPosts,open( "saveAllposts.p", "wb" ))

                num_posts = len(posts)
                counter += num_posts
                print(num_posts)
                print("Total urls collected so far: ", counter)
                output = word + "  " + str(num_posts) + "\n"
                tagnumresultref.write(output)

                for post in posts:
                    linepost = post.post_id + "  " + post.url + "\n"
                    fileref.write(linepost)
            else:
                continue

        except Exception as e:
            print("Problem: ", counter, "Exception is: ", e)

            # if(browserdriverB == None):
            #     print("chromedriver shut off")
            #     browserdriverB = webdriver.Chrome(exec_path)
            #     print("chrome webdriver restarted")
            # continue
    return allPosts

def main():
    en_dictionary = words.words()
    numwords_dictionary = len(en_dictionary)

    popular_tags = ["Startups", "Marketing", "Life",
                       "travel", "writing", "Tech",
                       "animals", "business", "entrepreneurship",
                       "design", "data", "science", "politics", "Motivation",
                       "Adventure", "Photography", "Books","Twitter",
                       "Facebook", "Instagram", "Environment",
                       "Reading", "medium"]

    filtered_words = [word for word in en_dictionary if word not in stopwords.words('english')]
    numwords_filtered_dictionary = len(filtered_words) #194 words filtered out

    inputnum = [random.randint(0,numwords_filtered_dictionary) for _ in range(10000)]
    searchwords = []

    for idx in inputnum:
        wordchoice = filtered_words[idx]
        searchwords.append(wordchoice)

    searchwords += popular_tags

    for word in searchwords:
        line = word + "\n"
        searchwordsref.write(line)


    allPosts = []
    searchTagsURLs = getPostsByTags(searchwords, 100, allPosts)

    fileref.close()
    tagnumresultref.close()
    searchwordsref.close()


main()
