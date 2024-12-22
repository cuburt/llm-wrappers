# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: somya.upadhyay, cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
from pinecone import Pinecone
from scripts.log import logger


class DB_Connector:
     
    def __init__(self, pinecone_api):
        self.pinecone_api= pinecone_api

    def connect_pinecone(self):
        pc_instance = None
        error_msg = ""
        try:
           pc_instance = Pinecone(api_key=self.pinecone_api)
        except Exception as ex:
            error_msg = f"Error while connecting pinecone , {ex}"
            logger.error(error_msg)

        return pc_instance, error_msg
          
    def get_pinecone_index(self, index_name="lotusscript-docs"):
        error_msg = None 
        index = None 
        try:
            pc_instance, error_msg = self.connect_pinecone()
            if not error_msg:
               index = pc_instance.Index(index_name) 
        except Exception as ex:
            error_msg = str(ex)
        return index, error_msg 
