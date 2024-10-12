from typing import List, Dict, Tuple
import requests
import json


class TimeoutException(Exception):
    pass


class PythonSandbox:
    def __init__(self):
        self.url = "http://34.145.156.214/sandboxes/python"

    def __call__(self, code):
        if "```python" in code:
            code = code.replace('```python', '')

        if "```" in code[len(code) - 5:]:
            code = code.replace('```', '')
        try:
            response = requests.post(self.url, data=json.dumps({"query": code})).json()
            print(response)
            out = response['response']
        except TimeoutException:
            out = "Timed out"

        except Exception as e:
            out = str(e)

        return out, code


class JavaScriptSandbox:
    def __init__(self):
        self.url = "http://34.145.156.214/sandboxes/javascript"

    def __call__(self, code):
        if "```js" in code:
            code = code.replace('```js', '')

        if "```javascript" in code:
            code = code.replace('```javascript', '')

        if "```" in code[len(code) - 5:]:
            code = code.replace('```', '')

        try:
            response = requests.post(self.url, data=json.dumps({"query": code})).json()
            print(response)
            out = response['response']
        except TimeoutException:
            out = "Timed out"

        except Exception as e:
            out = str(e)

        return out, code


class VoltScriptSandbox:
    def __init__(self):
        self.url = "http://34.145.156.214/sandboxes/voltscript"

    def __call__(self, code):
        if "```voltscript" in code:
            code = code.replace('```voltscript', '')

        if "```" in code[len(code) - 5:]:
            code = code.replace('```', '')

        try:
            response = requests.post(self.url, data=json.dumps({"query": code})).json()
            print(response)
            out = response['response']
        except TimeoutException:
            out = "Timed out"

        except Exception as e:
            out = str(e)

        return out, code