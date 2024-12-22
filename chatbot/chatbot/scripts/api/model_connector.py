# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:03:59 2023

@author: anilkumar.lenka, somya.upadhyay, cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
import asyncio
from scripts.log import logger
from scripts.modelpipeline import ModelPipeline


class ModelConnector:
    """Connects model operations to endppoint-router"""
    def __init__(self):
        self.model_pipeline_obj = ModelPipeline()
        #self.data_source_config = self.config["data_source"]

    def train_palm_model(self):
        """Reads data from data_source and trains the model.

        Returns:
            error_list(list) : list of error messages
        """
        errormsg = ""
        try:                   
            errormsg = self.model_pipeline_obj.generate_train_pipeline()
        except Exception as e:
            logger.error(str(e))  
            errormsg = str(e)

        return errormsg        
    
    def get_llm_predictions_batch(self, request, model, session_id):
        """Predict segments for a batch of user_ids

        Args:
            user_id_dict ([dict]): Dict of list of user_ids
            sample: {"user_id": [3155]}

        Returns:
            response: {"user_id": ....., "segment": [.........]}
        """
        try:
            query = request['query']
            enable_rag = request['enable_rag']
            result_data = self.model_pipeline_obj.generate_prediction(model, query, enable_rag, session_id)
            return result_data
        except Exception as e:
            raise Exception(str(e))

    def train_gemini_model(self):
        """Reads data from data_source and trains the model.

        Returns:
            error_list(list) : list of error messages
        """
        errormsg = ""
        try:                   
            errormsg = self.model_pipeline_obj.generate_train_pipeline()
        except Exception as e:
            logger.error(str(e))  
            errormsg = str(e)

        return errormsg
    
    def load_data(self):
        """Predict segments for a batch of user_ids

        Args:
            user_id_dict ([dict]): Dict of list of user_ids
            sample: {"user_id": [3155]}

        Returns:
            response: {"user_id": ....., "segment": [.........]}
        """
        errormsg = ""
        result_data = {}
        try:
            self.model_pipeline_obj = ModelPipeline()
            result_data = self.model_pipeline_obj.chunklen
            errormsg =self.model_pipeline_obj.errormsg
            if not result_data :
                print("no data to update")                
                
        except Exception as e:
            logger.error(str(e))  
            errormsg = str(e)
        
        return result_data, errormsg      
    
    def get_config(self):
        try:
            config_dic = self.model_pipeline_obj.config
            return config_dic
        except Exception as e:
            logger.error(str(e))
            return str(e)

    def update_config(self, new_config):
        response, error = "", ""
        if self.model_pipeline_obj.config != new_config:
            response, error = self.model_pipeline_obj.config_obj.upadate_config_details(new_config)
            self.model_pipeline_obj = ModelPipeline()
        # asyncio.run(self.model_pipeline_obj.ainitialize())
        return response, error
