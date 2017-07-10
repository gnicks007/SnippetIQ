from flask import render_template, Markup
from flask import request
from SnippetIQ import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2


import requests
import json
from bs4 import BeautifulSoup

import insightMedium_aws
from insightMedium_aws.parser import parse_user, parse_publication, parse_post, parse_single_post
from insightMedium_aws.constant import ROOT_URL, ACCEPT_HEADER, ESCAPE_CHARACTERS, COUNT
from insightMedium_aws.model import Sort

from SnippetIQ.a_model import ModelIt
from SnippetIQ.model import utility
from SnippetIQ.model import model_predict

##############################################################


@app.route('/')
@app.route('/index')

def index():
    return render_template("index.html")

@app.route('/bio')
def bio():
    return render_template("bio.html")

@app.route('/presentation')
def presentation():
    return render_template("presentation.html")

@app.route('/blog')
def blog():
    return render_template('index.html')

@app.route('/analysis')
def getUrl():
    error = "Please, enter a valid medium url."
    url = request.args.get('mediumurl')

    try:
        postObj = parse_single_post(url)
        postdataframe = model_predict.storePostinDf(postObj)
        tophighlights = model_predict.predictHighlight(postdataframe)
    except:
        return render_template("index.html", errorText=error)

    highlightedText = utility.highlight(url)
    return render_template("index.html", highlightedText=tophighlights)
    # return render_template("highlight.html", highlightedText=highlightedText)
