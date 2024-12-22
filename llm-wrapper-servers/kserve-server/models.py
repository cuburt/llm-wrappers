# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:23:21 2022

@author: Cuburt Balanon

@project: XAI

@input:

@output:

@des:
"""

import re
import kserve
from typing import Dict
import logging
from http import HTTPStatus
from fastapi.exceptions import HTTPException, RequestValidationError
from outputparser import SentimentParser, KeywordParser, ToneParser
from llm import PalmLLM


def format_query(query, sentiments=None, num_keys=None, tones=None):
    """
    Formats query with a template instruction.
    """
    formatted_output = ""
    if num_keys:
        parser = KeywordParser()
        formatted_output = parser.prompt.format(query=query, num_keys=str(num_keys))
    if sentiments:
        parser = SentimentParser()
        formatted_output = parser.prompt.format(query=query, sentiments=sentiments)
    if tones:
        parser = ToneParser()
        formatted_output = parser.prompt.format(query=query, tones=tones)

    return formatted_output


class ToneAnalysis(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        try:
            self.model = PalmLLM()
            self.ready = True

        except AssertionError as e:
            raise Exception("ERROR IN LOAD: " + str(e))

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:

        res = {}
        error_msg = ""
        output = ""

        try:
            default_tones = ['Sadness', 'Joy', 'Surprise', 'Disgust', 'Fear', 'Anger']
            assert payload.get("text") and isinstance(payload.get("text"), str) and len(
                payload.get("text")) > 5, "Invalid text in payload. Incorrect or missing value for text."
            text = payload.get("text")
            tones = payload.get('tones', default_tones)

            response_palm = self.model(format_query(query=text, tones=', '.join(tones)))
            parser = ToneParser()
            try:
                parsed_output = parser.parser.parse(response_palm)
            except:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", response_palm)
                parsed_output = parser.parser.parse(parsed_response_palm)

            output = [{"tone": t.tone, "score": t.score} for t in parsed_output.tones]

            res["predictions"] = output
            success_flag = True

        except AssertionError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

        except TypeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid text in payload. text must have a value of string datatype.")

        except RequestValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error in predict(): " + str(e))

        res["success"] = success_flag
        res["error"] = error_msg
        return res


class SentimentAnalysis(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        try:
            self.model = PalmLLM()
            self.ready = True

        except AssertionError as e:
            raise Exception("ERROR IN LOAD: " + str(e))

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:

        res = {}
        error_msg = ""
        output = ""

        try:
            default_sentiments = ["Highly Positive", "Positive", "Neutral", "Negative", "Highly Negative"]
            assert payload.get("text") and isinstance(payload.get("text"), str) and len(
                payload.get("text")) > 5, "Invalid text in payload. Incorrect or missing value for text."
            text = payload.get("text")
            sentiments = payload.get('sentiments', default_sentiments)

            response_palm = self.model(format_query(query=text, sentiments=', '.join(sentiments)))
            parser = SentimentParser()
            try:
                parsed_output = parser.parser.parse(response_palm)
            except:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", response_palm)
                parsed_output = parser.parser.parse(parsed_response_palm)

            output = [{"sentiment": s.sentiment, "score": s.score} for s in parsed_output.sentiments]

            res["predictions"] = output
            success_flag = True

        except AssertionError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

        except TypeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid text in payload. text must have a value of string datatype.")

        except RequestValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error in predict(): " + str(e))

        res["success"] = success_flag
        res["error"] = error_msg
        return res


class KeyphraseExtraction(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        try:
            self.model = PalmLLM()
            self.ready = True

        except AssertionError as e:
            raise Exception("ERROR IN LOAD: " + str(e))

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:

        res = {}
        error_msg = ""
        output = ""

        try:

            assert payload.get("text") and isinstance(payload.get("text"), str) and len(
                payload.get("text")) > 5, "Invalid text in payload. Incorrect or missing value for text."
            text = payload.get("text")
            num_keys = payload.get("num_keys", 5)

            response_palm = self.model(format_query(query=text, num_keys=num_keys))
            parser = KeywordParser()
            try:
                parsed_output = parser.parser.parse(response_palm)
            except:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", response_palm)
                parsed_output = parser.parser.parse(parsed_response_palm)
            output = [{"keyword": k.keyword, "score": k.score} for k in parsed_output.keywords]

            res["predictions"] = output
            success_flag = True

        except AssertionError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

        except TypeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid text in payload. text must have a value of string datatype.")

        except RequestValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error in predict(): " + str(e))

        res["success"] = success_flag
        res["error"] = error_msg
        return res

class Summariser(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        try:
            self.model = PalmLLM()
            self.ready = True

        except AssertionError as e:
            raise Exception("ERROR IN LOAD: " + str(e))

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:

        res = {}
        error_msg = ""
        output = ""

        try:

            assert payload.get("text") and isinstance(payload.get("text"), str) and len(payload.get("text")) > 5, "Invalid text in payload. Incorrect or missing value for text."
            text = payload.get("text")

            output = self.model("Provide a summary with about two sentences for the following article: " + text)

            res["predictions"] = output
            success_flag = True

        except AssertionError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

        except TypeError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Invalid text in payload. text must have a value of string datatype.")

        except RequestValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error in predict(): " + str(e))

        res["success"] = success_flag
        res["error"] = error_msg
        return res


if __name__ == "__main__":

    try:
        kserve.ModelServer().start([Summariser("summariser"),
                                    KeyphraseExtraction("keyphrase-extraction"),
                                    SentimentAnalysis("sentiment-analysis"),
                                    ToneAnalysis("emotion-analysis")])
    except Exception as e:
        logging.error(str(e))
