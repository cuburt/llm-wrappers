from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import re
import json
import subprocess
import platform



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
        _input = re.sub(r'http\S+', '', prompt.replace("\"", "'").replace("{", "").replace("}", "").replace("\\", "/"))
        cmd = ['/bin/bash', 'script.sh', '-t', _input, '-m', "chat", '-l', self._llm_type]

        if platform.system() == 'Windows':
            cmd = ['C:/Program Files/Git/bin/bash.exe', 'script.sh', '-t', _input, '-m', "chat", '-l', self._llm_type]

        # res = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE).stdout)
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
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
        _input = re.sub(r'http\S+', '', prompt.replace("\"", "'").replace("{", "").replace("}", "").replace("\\", "/"))
        cmd = ['/bin/bash', 'script.sh', '-t', _input, '-m', "chat", '-l', self._llm_type]

        if platform.system() == 'Windows':
            cmd = ['C:/Program Files/Git/bin/bash.exe', 'script.sh', '-t', _input, '-m', "chat", '-l', self._llm_type]

        # res = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE).stdout)
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
        response = res['predictions'][0]['content']

        return response