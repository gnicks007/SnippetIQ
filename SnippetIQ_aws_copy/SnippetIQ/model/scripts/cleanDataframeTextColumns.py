import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import spacy

columns_to_clean = ['post_content','sentence_text','subtitle', 'title',
       'tophighlight','post_tags']

def loadFile():
    with open("../../data/sentenceDataframe.p", "rb") as f:
        sentenceDataframe2 = pd.read_pickle(f)

    sentDataframe_noempty = sentenceDataframe2[~(sentenceDataframe2["sentence_text"]=="")]
    resetdf = sentDataframe_noempty.reset_index()
    del resetdf["index"]

    return resetdf


def clean(text):
    pstemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if not word.isdigit()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    #tokens = [pstemmer.stem(i) for i in tokens]
    tokens = [wordnet_lemmatizer.lemmatize(i) for i in tokens]
    tokens = [word.lower() for word in tokens]

    return tokens


def cleanPostContent(pdDataframe):
    numRows = pdDataframe.shape[0]
    # arr = [clean(pdDataframe.loc[idx, "post_content"]) for idx in range(0,numRows)]
    arr = [""]*numRows

    for idx in range(0, numRows):
        print(idx, end=" ")
        arr[idx] = clean(pdDataframe.loc[idx, "subtitle"])
        pickle.dump(arr, open( "../../data/cleanpost_subtitle.p", "wb" ))
    return arr


def main():
    df = loadFile()
    result = cleanPostContent(df)

main()
