from __future__ import print_function
import os

from insightMedium_aws.medium import Medium
from insightMedium_aws.model import Post, to_dict

# pandas / matplotlib
import pandas as pd
import numpy as np

# storage / persist-to-disk as objects
import pickle
# need to import hdf?

# String fuzzy matching
import Levenshtein #pip install python-levenshtein
import difflib


objKeys = ["post_id", "title", "post_date", "url",
          "recommend_count", "response_count", "read_time",
          "word_count", "image_count", "detectedLanguage",
          "post_content", "post_tags",
          "post_creatorId", "post_username",
          "blockquote_list", "tophighlight", "existTophighlight"]

mykeys = ["relLocationSentence", 'read_time',
       'sentence_num', 'image_count','post_word_count',
       'number_post_tags','post_number_sentences']

# populates dataframe for analysis
def populateDataframe(postObject):
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


def addColumnsToDataframe(dataframe):
    num_sentences_column = dataframe['post_content'].apply(lambda text: len(text.split('.')))
    dataframe['number_sentences'] = num_sentences_column

    dataframe["relLocationHighlight"] = 0
    dataframe[["relLocationHighlight"]] = dataframe[["relLocationHighlight"]].astype(float)

    dataframe["relLengthHighlight"] = 0
    dataframe[["relLengthHighlight"]] = dataframe[["relLengthHighlight"]].astype(float)

    dataframe["isHiglightInContent"] = np.nan
    return dataframe


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


def stringMatchOverlap(string1, string2):
    ratio = difflib.SequenceMatcher(None, string1, string2).ratio()
    return ratio

# Converts a dataframe storing posts to one with all tagged post sentences
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
    threshold = 0.8 #for fuzzy matching
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
    print("Works just fine")
