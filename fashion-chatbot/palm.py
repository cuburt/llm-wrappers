from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import json
import subprocess

# script = 'drive/MyDrive/llama/script.sh'
script = 'script.sh'
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
        print(prompt)
        res = json.loads(subprocess.run(['/bin/bash',
                                                   script,
                                                   '-t',
                                                   prompt.replace("\"", "'"),
                                                   '-m',
                                                   "chat",
                                                    '-l',
                                                    "palm",
                                                  #  "-c",
                                                  #  self.first_context,
                                                  #  "-e",
                                                  #  str(self.examples).replace("'", '"')
                                                   ],
                                           stdout=subprocess.PIPE).stdout)
        response = res['predictions'][0]['content']

        return response