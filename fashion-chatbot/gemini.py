from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import re
import json
import subprocess

# script = 'drive/MyDrive/llama/script.sh'
script = 'script.sh'
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
        res = json.loads(subprocess.run(['/bin/bash',
                                         script,
                                         '-t',
                                         re.sub(r'http\S+', '',
                                                prompt.replace("\"", "'").replace("{", "").replace("}", "").replace(
                                                    "\\", "/")),
                                         '-m',
                                         "chat",
                                         '-l',
                                         "gemini",
                                         #  "-c",
                                         #  self.first_context,
                                         #  "-e",
                                         #  str(self.examples).replace("'", '"')
                                         ],
                                        stdout=subprocess.PIPE).stdout)
        response = ''.join([r['candidates'][0]['content']['parts'][0]['text'] for r in res])

        return response