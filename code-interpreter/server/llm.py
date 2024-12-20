from typing import Any, List, Mapping, Optional, Dict
import json
import os
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class LLMRequest:
    def __call__(self, model_id:str, data:Dict):
        region = "us-central1"
        assert model_id in ["codechat-bison-32k", "gemini-pro", "text-bison", "multimodalembedding@001"], f"Sorry {model_id} is not currently supported."

        access_token = os.getenv("GEMINI_APIKEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={access_token}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url=url, headers=headers, data=json.dumps(data)).json()

        return response


class CodeyLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "codey"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        data = {
            "instances": [{"messages": [{"author": "user", "content": prompt}]}],
            "parameters": {"temperature": 0.1, "maxOutputTokens": 1024}
        }
        llm_request = LLMRequest()
        res = llm_request(model_id="codechat-bison-32k", data=data)
        response = res['predictions'][0]['candidates'][0]['content']

        return response


class GeminiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        print(prompt)
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": {"temperature": 0.1, "maxOutputTokens": 1024}
        }
        llm_request = LLMRequest()
        res = llm_request(model_id="gemini-pro", data=data)
        response = ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res])

        return response


class PalmLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "palm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        data = {
            "instances": [{"content": prompt}],
            "parameters": {"temperature": 0.1, "maxOutputTokens": 1024}
        }
        llm_request = LLMRequest()
        res = llm_request(model_id="text-bison", data=data)
        response = res['predictions'][0]['content']

        return response

    class MultimodalEmbeddingLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "multimodal-embedding"

        def _call(
                self,
                prompt: str,
                base64_encoded_img: str,
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> str:
            # if stop is not None:
            #     raise ValueError("stop kwargs are not permitted.")
            data = {
                "instances": [{
                    "text": prompt,
                    "image": {"bytesBase64Encoded": base64_encoded_img}
                }]
            }
            llm_request = LLMRequest()
            res = llm_request(model_id="multimodalembedding@001", data=data)
            response = res['predictions'][0]['content']

            return response

# class PalmLLM(LLM):
#     @property
#     def _llm_type(self) -> str:
#         return "palm"
#
#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any) -> str:
#         # if stop is not None:
#         #     raise ValueError("stop kwargs are not permitted.")
#         _input = re.sub(r'http\S+', '', prompt.replace("\"", "'").replace("{", "").replace("}", "").replace("\\", "/"))
#         cmd = ['/bin/bash', 'script.sh', '-t', _input, '-m', "chat", '-l', self._llm_type]
#
#         if platform.system() == 'Windows':
#             cmd = ['C:/Program Files/Git/bin/bash.exe', 'script.sh', '-t', _input, '-m', "chat", '-l', self._llm_type]
#
#         # res = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE).stdout)
#         sp = subprocess.run(cmd, capture_output=True, text=True)
#         if sp.stderr and not sp.stdout:
#             raise Exception(str(sp.stderr))
#         res = json.loads(sp.stdout)
#         response = res['predictions'][0]['content']
#
#         return response