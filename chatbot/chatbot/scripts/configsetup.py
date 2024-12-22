# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:19:15 2021

@author: anilkumar.lenka,somya.upadhyay

@des:
"""
from __future__ import print_function
import sys
import os
import json
from pathlib import Path
from deepmerge import Merger

from scripts.log import logger

PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)  # This is Project Root

# Set Base Path----------------------------------------------------------------
os.chdir(PROJECT_ROOT_DIR)
sys.path.append(PROJECT_ROOT_DIR)

class ConfigSetup:
    
    def __init__(self):
        pass

    def get_Merger(self):
        merger = Merger(
            [(list, ["override"]), (dict, ["merge"]), (set, ["union"])],
            ["override"],
            ["override"],
        )
        return merger

    def  get_config_details(self):
        error_message = ""
        configdic = {}
        try:
            with open(os.path.join(PROJECT_ROOT_DIR, "config.json")) as f:
                configdic = json.load(f)
            f.close()
        except Exception as e:
            logger.error(e)
            error_message = str(e)

        return configdic, error_message

    def upadate_config_details(self, Input_Config):
        error_message = ""
        data = {}
        try:
            assert Input_Config, "Configuration invalid."
            with open(os.path.join(PROJECT_ROOT_DIR, "config.json"), "w") as file:
                file.write(json.dumps(Input_Config))
            file.close()
            data, error_message = self.get_config_details()
            assert data == Input_Config, "Configuration update failed."
        except AssertionError as ae:
            logger.error(str(ae))
        except Exception as e:
            logger.error(str(e))
            error_message = str(e)
        
        return data, error_message
