# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: anilkumar.lenka, cuburt.balanon,manjunath_hegde

@project: XAI

@input:

@output:

@des
"""
from __future__ import print_function

from pathlib import Path
from scripts.server.chain import Chain
from scripts.configsetup import ConfigSetup
import threading
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent) #set project root directory


class ModelPipeline: 
    def __init__(self):
        self.chunklen = 0
        # moved config here to isolate references in the same file and reduce coupling
        self.config_obj = ConfigSetup()
        self.config, error = self.config_obj.get_config_details()
        self.chain = None
        self.ready = False
        self.ainitialize()

    def ainitialize(self):
        # Avoids TCP probe failure by immediate initialization of server with "unready" endpoints.
        initialize_thread = threading.Thread(target=self.initialize, daemon=True, name="initialize")
        initialize_thread.start()

    def initialize(self):
        self.chain = Chain(data_source=self.config["data_source"],
                           vectorstore_config=self.config["vectorstore_config"],
                           memory_window=self.config["memory_window"])
        self.ready = True

    @staticmethod
    def generate_train_pipeline():
        """Model Pipeline: Load Data, Preprocess Data, Form Causality Graph, Model Probability"""
        errormsg = ""

        return errormsg

    def generate_prediction(self, model, query, enable_rag=True, session_id=None):
        try:
            if not self.ready:
                raise Exception("Endpoint not ready. Please wait...")
            message_id, llm_response, source_documents = self.chain(query=query, llm=model, enable_rag=enable_rag, session_id=session_id)
            response = {"prediction": {"id": message_id, "answer": llm_response, "source_documents": source_documents}}
            return response
        except Exception as e:
            raise Exception(str(e))
