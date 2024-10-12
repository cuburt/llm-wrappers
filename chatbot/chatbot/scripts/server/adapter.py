# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon

@project: XAI

@input:

@output:

@des
"""

import requests
from flask import request
import json
from scripts.log import logger

class Adapter:

    def __init__(self):
        self.url = request.base_url

    def post(self, payload, headers=None, endpoint=''):
        try:
            if headers:
                result = requests.post(self.url + endpoint, headers=headers, data=payload)
            else:
                result = requests.post(self.url + endpoint, data=payload)
                
            return result.json()
        except Exception as e:
            logger.error(str(e)) 
            return {}

    def get(self, payload, headers, endpoint=''):
        try:
            result = requests.get(self.url + endpoint, headers=headers, data=payload)
            return result.json()
        except Exception as e:
            logger.error(str(e))
            return {}

class ArxivStoreAdapter(Adapter):
    #Need to shift config - Anil
    def __init__(self):
        self.host = "content-recommendation.default.example.com"
        self.url = "http://35.245.68.44/v1/models/ContentRecommendation"

    def build_headers(self):
        try:
            headers = {}
            if self.host:
                headers = {"host": self.host}
        except Exception as e:
            logger.error(str(e))    
        return headers

    def build_payload(self, title:str="", body:str=""):
        try:        
            payload = {"title": title,
                       "body": body}
            return json.dumps(payload)
        except Exception as e:
            logger.error(str(e))

class IQAgentAdapter(Adapter):

    def __init__(self):
        self.url = request.base_url

    def build_payload(self, enable_rag:bool=True, query:str=""):
        try:
            payload = {"enable_rag": enable_rag,
                       "query": query}
    
            return json.dumps(payload)
        except Exception as e:
            logger.error(str(e))