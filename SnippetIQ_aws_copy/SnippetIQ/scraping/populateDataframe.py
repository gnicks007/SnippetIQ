#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 07:59:15 2017

@author: Litombe
"""
from __future__ import print_function
import pandas as pd
import numpy as np


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