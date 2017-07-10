import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import nltk.data
import spacy
import os


#from scripts.cleanDataframeTextColumns import clean

def returnSentences(postObj):
    sentenceList = predictHighlight(df)

    msg = '''our current algorithm didn't find engaging Snippet to share.
        We are constantly improving our algorithm so stay tuned.'''

    if(sentenceList):
        return sentenceList
    else:
        return msg

#document is a dataframe of an article's sentences
def predictHighlight(sentenceDataframe):

    #Load trained random forest model from disk
    with open("rf_50estimators_balanced.pickle", "rb") as inputfile:
        modelrf = pd.read_pickle(inputfile)

    cols = ['relLocationSentence', 'read_time',
           'sentence_num', 'image_count','post_word_count',
           'number_post_tags','post_number_sentences']

    numericDf = sentenceDataframe[cols]
    ypredict_doc = modelrf.predict(numericDf)
    result = modelrf.predict_proba(numericDf)
    arr = result[:,1]
    recommend = arr.argsort()[-3:][::-1]

    highlights = []
    for i in recommend:
        sent = sentenceDataframe.loc[i, "sentence"]
        highlights.append(sent)

    return highlights
    #print(modelrf.predict_proba(numericDf))


def storePostinDf(postobject):
# postobject_keys = ["post_id", "title", "post_date", "url",
#           "recommend_count", "response_count", "read_time",
#           "word_count", "image_count", "detectedLanguage",
#           "post_content", "post_tags",
#           "post_creatorId", "post_username",
#           "blockquote_list", "tophighlight", "existTophighlight"]
    colNames = ["relLocationSentence", 'read_time',
           'sentence_num', 'image_count','post_word_count',
           'number_post_tags','post_number_sentences', "sentence"]

    numcols = len(colNames)
    numericCols = colNames[0:numcols-1]
    # Store numeric values
    postdf = pd.DataFrame(columns=colNames)
    postdf[numericCols] = postdf[numericCols].astype(float)
    #post["sentence"] = postdf["sentence"].astype(object)

    content = postobject.post_content

    # sentence tokenization using nltk
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceList = sent_detector.tokenize(content.strip())

    # sentence tokenization using a naive split
    #sentenceList = content.split('.') #replace this with proper sentence tokenization

    # sentence tokenization with spacy
    # nlp = spacy.load('en')
    # doc = nlp(content.strip())
    # sentenceList = [sent.string.strip() for sent in doc.sents]



    numSentences = len(sentenceList)



    for idx, sent in enumerate(sentenceList):
        postdf.loc[idx, "sentence_num"] = idx
        postdf.loc[idx, "relLocationSentence"] = idx/numSentences
        postdf.loc[idx, "sentence"] = sent
        postdf.loc[idx,'read_time'] = postobject.read_time
        postdf.loc[idx, 'image_count'] = postobject.image_count
        postdf.loc[idx, 'number_post_tags'] = len(postobject.post_tags)
        postdf.loc[idx, 'post_word_count'] = postobject.word_count
        postdf.loc[idx, 'post_number_sentences'] = numSentences

    return postdf


def cleanPostContent(pdDataframe, columnName):
    numRows = pdDataframe.shape[0]

    for idx in range(0, numRows):
        cleanedString = clean(pdDataframe.loc[idx, columnName])
        finalString = " ".join(cleanedString)
        pdDataframe.loc[idx, columnName] = finalString

    return pdDataframe

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
