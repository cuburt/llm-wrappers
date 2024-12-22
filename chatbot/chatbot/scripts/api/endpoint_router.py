"""
Created on Thu Jun 22 10:03:59 2023

@author: anilkumar.lenka

@project: XAI

@input:

@output:

@des
"""
import json
from typing import Optional
from fastapi import APIRouter, Query
from fastapi import Request
import csv
from scripts.log import logger
from scripts.api.model_connector import ModelConnector


router = APIRouter()
class Item():
    data_path: str
    data_types: list

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data_path": "./data/",
                    "data_types": ["csv","pdf","html"],
                }
            ]
        }
    }

model_connector_obj = ModelConnector()
def add_column_to_df(query, data_res):
    # Open the CSV file in append mode
    with open('test_data.csv', 'a', newline='') as csvfile:
        # Define the fieldnames for the CSV
        fieldnames = ['query', 'data_res']
        # Create a CSV DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        # Write the data to the CSV file
        writer.writerow({'query': query, 'data_res': data_res})


@router.post("/get_{model}_answer_generate", tags=["xaiiqbot"])
def get_llm_generate(model: str, request: dict, session_id: Optional[str] = Query(None)):
    try:
        assert len(request.values()) != 0, "No entity found"
        is_success = True
        response_dict = model_connector_obj.get_llm_predictions_batch(request, model, session_id)
        error_message = ""
    except Exception as e:
        logger.error(str(e))
        response_dict = {}
        is_success = False
        error_message = str(e)

    data_res = {"success": is_success,
                "data": response_dict,
                "errors": error_message}

    return data_res


@router.get("/{model}_train_model", tags=["xaiiqbot"])
def train_model(model: str, request: Request):
    """API route to train a model

    Args:
        request (Request): _description_

    Returns:
        response (dict): JSON respopnse
    """
    is_success = False
    errormsg = ''
    data_res = {}
    try:
        errormsg = eval(f"model_connector_obj.train_{model}_model()")  # model_trigger.main()
    except Exception as e:
        errormsg = str(e)
        logger.error("Error %s", e)

    if len(errormsg) == 0:
        is_success = True

    data_res = {"success": is_success, "errors": errormsg}
    return data_res

@router.post("/get_gemini_answer_generate", tags=["xaiiqbot"])
def get_gemini_generate(query: dict, request: Request):    
    data_res = {}
    response_dict = {}
    is_success = False
    error_message = ""
    if len(query.values()) == 0:
        error_message = "No entity found"

    if len(error_message) == 0:
        try:
            text, error_message = model_connector_obj.get_gemini_predictions_batch(
                query)
            text=text.replace("\\n",'').replace('\\','')
            text=text.replace("{\"",'{~')
            text=text.replace("[\"",'[~')
            text=text.replace("(\"",'(~')
            text=text.replace("\":",'~:')
            text=text.replace(": \"",': ~')
            text=text.replace("\", \"",'~, ~')
            text=text.replace("\"}",'~}')
            text=text.replace("\"]",'~]')
            text=text.replace("\")",'~)')
            text=text.replace("\"",'`')
            text=text.replace("~",'\"')
            
            response_dict=json.loads(text)
            if len(error_message) == 0:
                is_success = True
        except Exception as err:
            error_message = err
            logger.error("Error %s", error_message)

    data_res = {"success": is_success,
                "data": response_dict,
                "errors": error_message}
    add_column_to_df(query,data_res)

    return data_res

@router.get("/palm_train_generate_code", tags=["xaiiqbot"])
def palm_train_generate_code(request: Request):
    """API route to train a model

    Args:
        request (Request): _description_

    Returns:
        response (dict): JSON respopnse
    """
    is_success = False
    errormsg = ''
    data_res = {}
    try:
        errormsg = model_connector_obj.train_palm_generate_code()  # model_trigger.main()
    except Exception as e:
        errormsg = str(e)
        logger.error("Error %s", e)

    if len(errormsg) == 0:
        is_success = True

    data_res = {"success": is_success, "errors": errormsg}
    return data_res

@router.post("/get_palm_generate_code", tags=["xaiiqbot"])
def get_palm_generate_code(query: dict, request: Request):    
    data_res = {}
    response_dict = {}
    is_success = False
    error_message = ""
    if len(query.values()) == 0:
        error_message = "No entity found"

    if len(error_message) == 0:
        try:
            text, error_message = model_connector_obj.get_palm_predictions_generate_code_batch(
                query)
            text=text.replace("\\n",'').replace('\\','')
            response_dict=json.loads(text)
            if len(error_message) == 0:
                is_success = True
        except Exception as err:
            error_message = err
            logger.error("Error %s", error_message)

    data_res = {"success": is_success,
                "data": response_dict,
                "errors": error_message}

    return data_res


@router.get("/gemini_train_generate_code", tags=["xaiiqbot"])
def train_model(request: Request):
    """API route to train a model

    Args:
        request (Request): _description_

    Returns:
        response (dict): JSON respopnse
    """
    is_success = False
    errormsg = ''
    data_res = {}
    try:
        errormsg = model_connector_obj.train_gemini_generate_code()  # model_trigger.main()
        is_success = True
    except Exception as e:
        errormsg = str(e)
        logger.error("Error %s", e)

    if len(errormsg) == 0:
        is_success = True

    data_res = {"success": is_success, "errors": errormsg}
    return data_res

@router.post("/get_gemini_generate_code", tags=["xaiiqbot"])
def get_gemini_generate_code(query: dict, request: Request):    
    data_res = {}
    response_dict = {}
    is_success = False
    error_message = ""
    if len(query.values()) == 0:
        error_message = "No entity found"

    if len(error_message) == 0:
        try:
            text, error_message = model_connector_obj.get_gemini_predictions_generate_code_batch(
                query)
            text=text.replace("\\n",'').replace('\\','')
            response_dict=json.loads(text)
            if len(error_message) == 0:
                is_success = True
        except Exception as err:
            error_message = err
            logger.error("Error %s", error_message)

    data_res = {"success": is_success,
                "data": response_dict,
                "errors": error_message}

    return data_res

@router.get("/load_data", tags=["xaiiqbot"])
def load_data(request: Request): 
    response_dict = {}
    is_success = False
    error_message = ""
    if len(error_message) == 0:
        try:
            response_dict, error_message = model_connector_obj.load_data()
            if not error_message:
                is_success = True
        except Exception as err:
            error_message = err
            logger.error("Error %s", error_message)
    data_res = {"success": is_success,
                "data": f"{response_dict} chunks loaded",
                "errors": error_message}
    return data_res

@router.get("/getconfig", tags=["xaiiqbot"])
def get_config(request: Request):
    """API route to get config

    Args:
        request (Request): _description_

    Returns:
        response (dict): JSON respopnse
    """
    # model_connector_obj = ModelConnector()
    is_success = False
    errormsg = ''
    config = {}
    
    try:
        config = model_connector_obj.get_config()
    except Exception as e:
        logger.error(str(e))
        errormsg = str(e)

    if len(errormsg) == 0:
        is_success = True

    data_res = {"success": is_success, "config": config, "errors": errormsg}
    return data_res

@router.post("/updateconfig", tags=["xaiiqbot"])
def update_config(new_config: dict, request: Request):
    success=False
    error=""
    response={}
    try:
        response, error = model_connector_obj.update_config(new_config)
        if not error:
            success = True
    except Exception as e:
        error = str(e)
        logger.error(error)
    return {"success": success, "new_config": response, "error": error}
