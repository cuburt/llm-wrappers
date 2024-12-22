# -*- coding: utf-8 -*-
"""

@author: cuburt.balanon,manjunath_hegde

@project: XAI

@input:

@output:

@des
"""

from typing import Any, List, Mapping, Optional, Dict
import json
import subprocess
import platform
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pathlib import Path
import os
from scripts.log import logger
import time

PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent)  # set project root directory


class GoogleRequest:
    def __call__(self, model_id: str, data: Dict):
        try:
            google_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemma-2-27b-it"]
            if model_id not in google_models:
                return f"Sorry {model_id} is not currently supported."
            access_token = os.getenv("GEMINI_APIKEY")
            if not access_token:
                return "Error 401: No Google API-KEY saved."
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={access_token}"
            headers = {"Content-Type": "application/json"}
            response = requests.post(url=url, headers=headers, data=json.dumps(data)).json()
            return response

        except Exception as e:
            logger.error(str(e))


class NvidiaRequest:
    def __call__(self, model_id: str, data: Dict):
        try:
            nvidia_models = ["meta/llama3-70b-instruct"]
            if model_id not in nvidia_models:
                return f"Sorry {model_id} is not currently supported."

            url = "https://integrate.api.nvidia.com/v1/chat/completions"

            api_key_path = "scripts/server/.nvidia_api_key.txt"
            if platform.system() == 'Windows':
                api_key_path = api_key_path.split('/')
                api_key_path = os.path.join(PROJECT_ROOT_DIR, *api_key_path)
            else:
                api_key_path = os.path.join(PROJECT_ROOT_DIR, api_key_path)
            with open(api_key_path, 'r') as f:
                access_token = f.read().rstrip()
            if not access_token:
                return "Error 401: No Nvidia API-KEY saved."

            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
            response = requests.post(url=url, headers=headers, data=json.dumps(data)).json()
            return response

        except Exception as e:
            logger.error(str(e))


class OpenAIRequest:
    def __call__(self, model_id: str, data: Dict):
        try:
            openai_models = ["gpt-3.5", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            if model_id not in openai_models:
                return f"Sorry {model_id} is not currently supported."

            url = "https://api.openai.com/v1/chat/completions"

            api_key_path = "scripts/server/.openai_api_key.txt"
            if platform.system() == 'Windows':
                api_key_path = api_key_path.split('/')
                api_key_path = os.path.join(PROJECT_ROOT_DIR, *api_key_path)
            else:
                api_key_path = os.path.join(PROJECT_ROOT_DIR, api_key_path)
            with open(api_key_path, 'r') as f:
                access_token = f.read().rstrip()
            if not access_token:
                return "Error 401: No OpenAI API-KEY saved."

            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
            response = requests.post(url=url, headers=headers, data=json.dumps(data)).json()
            return response

        except Exception as e:
            logger.error(str(e))


class GeminiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-1.0-pro-002"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            # print(prompt)
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {"temperature": 1, "maxOutputTokens": 2048, "topP": 1}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if len(res) >= 1 and 'error' in res[0] and 'code' in res[0]['error'] and 'message' in res[0]['error']:
                return f"Error {res[0]['error']['code']}: {res[0]['error']['message']}"
            elif len(res) >= 1 and 'candidates' in res[0]:
                return ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res])
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class Gemini15LLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-1.5-pro-001"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            # print(prompt)
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {"temperature": 1, "maxOutputTokens": 2048, "topP": 1}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if len(res) >= 1 and 'error' in res[0] and 'code' in res[0]['error'] and 'message' in res[0]['error']:
                return f"Error {res[0]['error']['code']}: {res[0]['error']['message']}"
            elif len(res) >= 1 and 'candidates' in res[0]:
                return ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res if len(r['candidates']) > 0 and "content" in r['candidates'][0] and len(r['candidates'][0]['content']['parts']) > 0 and "text" in r['candidates'][0]['content']['parts'][0]])
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class GeminiFlashLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-1.5-flash-001"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            # print(prompt)
            data = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {"temperature": 1, "maxOutputTokens": 2048, "topP": 1}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if len(res) >= 1 and 'error' in res[0] and 'code' in res[0]['error'] and 'message' in res[0]['error']:
                return f"Error {res[0]['error']['code']}: {res[0]['error']['message']}"
            elif len(res) >= 1 and 'candidates' in res[0]:
                return ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res if len(r['candidates']) > 0 and "content" in r['candidates'][0] and len(r['candidates'][0]['content']['parts']) > 0 and "text" in r['candidates'][0]['content']['parts'][0]])
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class PalmLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "text-bison"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "instances": [{"content": prompt}],
                "parameters": {"temperature": 0.1, "maxOutputTokens": 1024}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if 'error' in res and 'code' in res['error'] and 'message' in res['error']:
                return f"Error {res['error']['code']}: {res['error']['message']}"
            elif 'predictions' in res and len(res['predictions']) > 0:
                return res['predictions'][0]['content']
            else:
                return res(res)
        except Exception as e:
            logger.error(str(e))


class CodeyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "codechat-bison-32k"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "instances": [{"messages": [{"author": "user", "content": prompt}]}],
                "parameters": {"temperature": 0.1, "maxOutputTokens": 1024}
            }
            llm_request = GoogleRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if 'error' in res and 'code' in res['error'] and 'message' in res['error']:
                return f"Error {res['error']['code']}: {res['error']['message']}"
            elif 'predictions' in res and len(res['predictions']) > 0:
                return res['predictions'][0]['candidates'][0]['content']
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class Llama3LLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "meta/llama3-70b-instruct"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "model": "meta/llama3-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1024,
                "top_p": 0.7,
                "stream": False
            }
            llm_request = NvidiaRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if 'choices' in res and len(res['choices']) > 0:
                return res['choices'][0]['message']['content']
            else:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class GPT4oLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gpt-4o"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1,
                "top_p": 1,
                "stream": False
            }
            llm_request = OpenAIRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if res and 'choices' in res and len(res['choices']) > 0:
                return res['choices'][0]['message']['content']
            elif res:
                return str(res)

        except Exception as e:
            logger.error(str(e))


class GPT4oMiniLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gpt-4o-mini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        try:
            data = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1,
                "top_p": 1,
                "stream": False
            }
            llm_request = OpenAIRequest()
            start = time.time()
            res = llm_request(model_id=self._llm_type, data=data)
            logger.info(f"LLM RT: {(time.time() - start):.4f}")
            if res and 'choices' in res and len(res['choices']) > 0:
                return res['choices'][0]['message']['content']
            elif res:
                return str(res)

        except Exception as e:
            logger.error(str(e))
