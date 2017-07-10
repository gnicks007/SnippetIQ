from __future__ import print_function
import os

from insightMedium_aws.medium import Medium
from insightMedium_aws.model import Post, to_dict


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

#%matplotlib inline


mykeys = ["post_id", "title", "post_date", "url",
          "recommend_count", "response_count", "read_time",
          "word_count", "image_count", "detectedLanguage",
          "post_content", "post_tags",
          "post_creatorId", "post_username",
          "blockquote_list", "tophighlight", "existTophighlight"]




# input: Load filename, a csv file into a pandas dataframe
# output: return pandas dataframe with filename data
def loadFileIntoPandas(filename):
    dataframe = pd.read_csv(filename, sep=" ", header=None)

    return dataframe

# TO DO -
def populateDatabase(pdFrame, dbname, username):

    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    print(engine.url)

    if not database_exists(engine.url):
        create_database(engine.url)
    print(database_exists(engine.url))

    table = dbname
    pdFrame.to_sql(table, engine, if_exists='replace')

    return engine, table  #return the connection engine, and the databasename

# input: postObjectList is an array of Post objects for medium articles
# output: return a pandas dataframe with columns populated by appropriate keywords
# usecase: specifically for medium posts. Relies on knowing dict-keys
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

# input: article is text to be summarized
# result is an array with [luhn, lex, lsa] results
# usecase: general
def models_LUHN_LEX_LSA(article):
    ##    Candidate models:
        #        Bag of Words
        #        FastText
        #        word2vec
        #        LDA (topic extraction)
        #        skip-thoughts
        #        doc2vec
        #        LSTM

    LANGUAGE = "english"
    stop = get_stop_words(LANGUAGE)
    stemmer = Stemmer(LANGUAGE)
    parser = PlaintextParser.from_string(article, Tokenizer(LANGUAGE))

    result = []

    summarizerLUHN = LUHN(stemmer)
    summarizerLUHN.stop_words = stop

    summarizerLEX = LEX(stemmer)
    summarizerLEX.stop_words = stop

    summarizerLSA = LSA(stemmer)
    summarizerLSA.stop_words = stop

    LUHNsentence = summarizerLUHN(parser.document, 1) #summarize document with one sentence
    LEXsentence = summarizerLEX(parser.document, 1) #summarize document with one sentence
    LSAsentence = summarizerLSA(parser.document, 1) #summarize document with one sentence

    for sentence in LUHNsentence:
        LUHNsummary = sentence
    for sentence in LEXsentence:
        LEXsummary = sentence
    for sentence in LSAsentence:
        LSAsummary = sentence

    result.append(LUHNsummary)
    result.append(LEXsummary)
    result.append(LSAsummary)

    return result

# dataframe: input with post_content column to summarize.
# returns summary text results in inplace under columns ["LUHN", "LEX", "LSA"]
# usecase: general
def models_LUHN_LEX_LSA_2(dataframe):
    LANGUAGE = "english"
    stop = get_stop_words(LANGUAGE)
    size = len(dataframe)
    stemmer = Stemmer(LANGUAGE)

    for i in range(0, size):
        article = dataframe.loc[i, "post_content"]

        parser = PlaintextParser.from_string(article, Tokenizer(LANGUAGE))

        summarizerLUHN = LUHN(stemmer)
        summarizerLUHN.stop_words = stop

        summarizerLEX = LEX(stemmer)
        summarizerLEX.stop_words = stop

        summarizerLSA = LSA(stemmer)
        summarizerLSA.stop_words = stop

        LUHNsentence = summarizerLUHN(parser.document, 1) #summarize document with one sentence
        LEXsentence = summarizerLEX(parser.document, 1) #summarize document with one sentence
        LSAsentence = summarizerLSA(parser.document, 1) #summarize document with one sentence

        for sentence1 in LUHNsentence:
            LUHNsummary = sentence1
        for sentence2 in LEXsentence:
            LEXsummary = sentence2
        for sentence3 in LSAsentence:
            LSAsummary = sentence3

        dataframe.loc[i, "LUHN"] = LUHNsummary
        dataframe.loc[i, "LEX"] = LEXsummary
        dataframe.loc[i, "LSA"] = LSAsummary

# dataframe: input with post_content column to summarize.
# returns summary text results in inplace under column ["gensim"]
def gensimSummarize(dataframe):
    size = len(dataframe)

    for i in range(0,size):
        try:
            ans = dataframe.loc[i, "post_content"]
            res = summarize(ans, ratio=0.01) #can modify ratio of summarizing text length to total text
            dataframe.loc[i,"gensim"] = res
        except ValueError:
            print(i)
            dataframe.loc[i,"gensim"] = ''
        except TypeError:
            print(i)
            dataframe.loc[i,"gensim"] = ''


#TO DO
# Find which split of "stringBeforeSplit", highlightText falls in
def findHighlightsIndex(stringBeforeSplit, highlightText):
    return True

#TO DO
# filename could have .p extension
def pickleDataframe(dataframe, filename):
    pickle.dump(dataframe, open(filename, "wb" ))


def saveDataframeToHDF(dataframe, filename):
    dataframe.to_hdf(filename, "w")
    return dataframe

#input: loads a pickle file: picklefilename
#output: a python object(s) stored in the pickle file
# usecase: general
def pickleIntoPythonObject(picklefilename):
    with open(picklefilename, 'rb') as f:
        objfile = pickle.load(f)

    return objfile

#input: loads a pickle file: picklefilename
#output: a python object(s) stored in the pickle file
#usecase: specifically for medium posts. Relies on knowing dict-keys
def picklefileIntoPandas(picklefilename):
    obj = pickleIntoPythonObject(picklefilename)
    df = populateDataframe(obj)

    return df

# input: dataframe with medium post contents
# output: returns dataframe rows in english and with highlights and no duplicates
# output: adds other columns
# usecase: assumes Medium post contents
def dataframeEnglishWithHighlight(dataframe):
    df_no_duplicates = dataframe.drop_duplicates('post_id')
    df_english_noduplicates = df_no_duplicates[df_no_duplicates["detectedLanguage"]=='en']

    data = df_english_noduplicates.reset_index()
    del df_english_noduplicates["index"] #drop extra column

    return data

# input: dataframe with medium post contents
# output: adds columns to dataframe
# output: number_sentences, reLlocationHighlight, relLengthHighlight, isHighlightInContent
# output:
# usecase: assumes Medium post contents, and that dataframe doesn't need to be reindexed
def addColumnsToDataframe(dataframe):
    num_sentences_column = dataframe['post_content'].apply(lambda text: len(text.split('.')))
    dataframe['number_sentences'] = num_sentences_column

    dataframe["relLocationHighlight"] = 0
    dataframe[["relLocationHighlight"]] = dataframe[["relLocationHighlight"]].astype(float)

    dataframe["relLengthHighlight"] = 0
    dataframe[["relLengthHighlight"]] = dataframe[["relLengthHighlight"]].astype(float)

    dataframe["isHiglightInContent"] = np.nan
    return dataframe

# input: dataframe with medium post contents
# output: plots of feature distributions and averages
# usecase: assumes Medium post contents, and has added Columns
def plotVariableDistributions(dataframe):
    dataframe .hist(column="read_time", bins=50, alpha=0.4)
    print("Maximum Read time: ", dataframe['read_time'].max())

    dataframe .hist(column="image_count", color='g', bins=50, alpha=0.4)
    print("mean images: ", dataframe['image_count'].mean())

    dataframe.hist(column="number_blockquotes", bins=50, alpha=0.4)
    print("mean number of blockquotes: ", dataframe['number_blockquotes'].mean())

    dataframe.hist(column="word_count", bins=50, alpha=0.4)
    print("mean number of word_count: ", dataframe['word_count'].mean())

    dataframe.hist(column="response_count", bins=50, alpha=0.4)
    print("mean number of response_count: ", dataframe['response_count'].mean())

    dataframe.hist(column="recommend_count", bins=50, alpha=0.4)
    print("mean number of recommend_count: ", dataframe['recommend_count'].mean())

    dataframe.hist(column="number_post_tags", bins=50, alpha=0.4)
    print("mean number of number_post_tags: ", dataframe['number_post_tags'].mean())

    dataframe.hist(column="number_sentences", bins=50, alpha=0.4)
    print("mean number of sentences: ", dataframe['number_sentences'].mean())

    return


# Calculate relativeLocation and relativeLength of tophighlight
# output: relativeLocation, relativeLength, readtime
# some posts want match ignore for now. remove unicode from post_content
def relLocationAndLengthTophighlights(dataframe):
    x = [] #relativeLocation
    y = [] #relativeLength
    r = [] #read time

    size = len(dataframe)
    for i in range(0,size):
        try:
            article = dataframe.loc[i, "post_content"]
            highlight = dataframe.loc[i, "tophighlight"]
            readtime = dataframe.loc[i, "read_time"]

            totalLength = len(article)
            length = len(highlight)
            relLength = length/totalLength


            assert(highlight in article)
            location = article.find(highlight)

            relLocation = location/totalLength
            x.append(relLocation)
            y.append(relLength)
            r.append(readtime)

            dataframe.loc[i, "relLengthHighlight"] = relLength
            dataframe.loc[i, "relLocationHighlight"] = relLocation
            dataframe.loc[i, "isHiglightInContent"] = True
        except AssertionError:
            #illustrates that when this fails it's due to presence of unicode characters
            print(i, " ", all(ord(char) < 128 for char in highlight))

            #If it fails try and remove emojies etc.
            #How to remove emoji's, unicode and other characters from text
            #maybe use beautiful soup parser to take care of that.
            #Create new pandas dataframe with cleaned up strings

    # Plot: x - relLocation, y - relLength, r - readtime
    # plt.scatter(x, y, s=r, alpha=0.5)
    # plt.scatter(y, r, alpha=0.2)
    # plt.scatter(x, r, alpha=0.2)
    # plt.scatter(r, x, alpha=0.2)
    return x, y, r



# Find out which text summarizations match the top highlight in a dataframe
def countSummarizationMatches(dataframe):

    size = len(dataframe)

    for i in range(0,size):
        highlight = dataframe.loc[i, "tophighlight"]
        summaryLUHN = dataframe.loc[i, "LUHN"]
        summaryLEX = dataframe.loc[i, "LEX"]
        summaryLSA = dataframe.loc[i, "LSA"]
        summarygensim = dataframe.loc[i, "gensim"]

        lenLuhn = len(summaryLUHN)
        lenLEX = len(summaryLEX)
        lenLSA = len(summaryLSA)
        lengensim = len(summarygensim)

        test1 = highlight.find(summaryLUHN)
        test2 = highlight.find(summaryLUHN)
        test3 = highlight.find(summaryLUHN)
        test4 = highlight.find(summarygensim)

        #if the functions return negative then nothing was found. otherwise, it's a substring
        #also tests for empty strings for the summary algorithm strings
        if(test1<0 or lenLuhn==0):
            dataframe.loc[i, "LUHNcount"] = 0
        else:
            dataframe.loc[i, "LUHNcount"] = 1

        if(test2<0 or lenLEX==0):
            dataframe.loc[i, "LEXcount"] = 0
        else:
            dataframe.loc[i, "LEXcount"] = 1

        if(test3<0 or lenLSA==0):
            dataframe.loc[i, "LSAcount"] = 0
        else:
            dataframe.loc[i, "LSAcount"] = 1

        if(test4<0 or lengensim==0):
            dataframe.loc[i, "gensimcount"] = 0
        else:
            dataframe.loc[i, "gensimcount"] = 1


    luhncount = dataframe["LUHNcount"].sum()
    lexcount = dataframe["LEXcount"].sum()
    lsacount = dataframe["LSAcount"].sum()
    gencount = dataframe["gensimcount"].sum()

    print(luhncount, " ", lexcount, " ", lsacount, " ", gencount)



def stringMatchOverlap(string1, string2):
    ratio = difflib.SequenceMatcher(None, string1, string2).ratio()
    return ratio

# Converts a dataframe storing posts to one with all tagged post sentences
# IMPORTANT function: Can use for different vectorization, model & plot schemes
#highly shared post is 260 rows
def toSentenceDataFrame(dataframe):
    size = len(postDataframe)
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
        rowitem = postDataframe.loc[idx, "post_content"]
        highlight = postDataframe.loc[idx, "tophighlight"]

        highlightArray = highlight.split(".")
        rowcontentArray = rowitem.split(".")

        messageLength = len(rowcontentArray)
        numHighlights = len(highlightArray) #this is the number of sentences associated with each post highlight

        totSentences+=messageLength
        totHighlights+=numHighlights

        for c, content in enumerate(rowcontentArray):

            newindex+=1
            senlength = len(content)

            #code can be refactored by creating a dict of keys e.g. {"post_number_sentences": "number_sentences"}
            #or loop through keys stored in two arrays
            df.loc[newindex, "post_date"] = postDataframe.loc[idx, "post_date"]
            df.loc[newindex, "post_id"] = postDataframe.loc[idx, "post_id"]
            df.loc[newindex, "post_num"] = idx
            df.loc[newindex, "sentence_num"] = c
            df.loc[newindex, "sentence_text"] = content
            df.loc[newindex, "isHighlight"] = "no"
            df.loc[newindex, "read_time"] = postDataframe.loc[idx, "read_time"]
            df.loc[newindex, "relLocationHighlight"] = postDataframe.loc[idx, "relLocationHighlight"]
            df.loc[newindex, "relLengthHighlight"] = postDataframe.loc[idx, "relLengthHighlight"]
            df.loc[newindex, "relLocationSentence"] = (c+1)/messageLength
            df.loc[newindex, "relLengthSentence"] = senlength
            df.loc[newindex, "post_word_count"] = postDataframe.loc[idx, "word_count"]
            df.loc[newindex, "image_count"] = postDataframe.loc[idx, "image_count"]
            df.loc[newindex, "subtitle"] = postDataframe.loc[idx, "subtitle"]
            df.loc[newindex, "title"] = postDataframe.loc[idx, "title"]
            df.loc[newindex, "post_tags"] = postDataframe.loc[idx, "post_tags"]
            df.loc[newindex, "response_count"] = postDataframe.loc[idx, "response_count"]
            df.loc[newindex, "recommend_count"] = postDataframe.loc[idx, "recommend_count"]
            df.loc[newindex, "number_post_tags"] = postDataframe.loc[idx, "number_post_tags"]
            df.loc[newindex, "url"] = postDataframe.loc[idx, "url"]
            df.loc[newindex, "post_number_sentences"] = postDataframe.loc[idx, "number_sentences"]
            df.loc[newindex, "post_content"] = postDataframe.loc[idx, "post_content"]
            df.loc[newindex, "tophighlight"] = postDataframe.loc[idx, "tophighlight"]

            print(newindex)
            for h, highlite in enumerate(highlightArray):
                #print(newindex)
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
    pickledatafile = "finalscrapePostings_final copy.p"
    with open(pickledatafile, 'rb') as f:
        objfile = pickle.load(f)
