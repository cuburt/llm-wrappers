from typing import List, Dict, Tuple
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from llm import CodeyLLM, PalmLLM, GeminiLLM  # IMPORTANT! Do not remove.
from output_parser import CodeInterpreterSchema
from prompt_template import CodeInterpreterPromptTemplate, CodeAssistantPromptTemplate
import re
from vector_index import VectorStoreFactory
from retriever import RetrieverFactory
from sandbox import PythonSandbox, JavaScriptSandbox, VoltScriptSandbox


class BasePipeline:
    def __init__(self, docs: List[Document], model, prompt_template):
        self.llm = eval(f"{model.capitalize()}LLM()")
        self.vectorstore = VectorStoreFactory(docs).vectorstore
        self.retriever = RetrieverFactory(vectorstore=self.vectorstore, k=3, search_type="similarity")
        self.chain = self.build_chain(self.llm, prompt_template, self.retriever)
        self.python_sandbox = PythonSandbox()
        self.javascript_sandbox = JavaScriptSandbox()
        self.voltscript_sandbox = VoltScriptSandbox()

    @staticmethod
    def build_chain(llm, prompt_template, retriever) -> RetrievalQA:
        """
        retriever's constructor
        """
        return RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            prompt=prompt_template,
            return_source_documents=True
        )

    def run_code(self, code_response, timeout: float = 3.0, tmp_dir: str = None) -> Tuple[str, str, str, str]:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem.
        """
        print(code_response)
        exec_out = ""
        code = ""
        lang = ""
        if code_response and code_response != "" and "python" in code_response:
            lang = "python"
            exec_out, code = self.python_sandbox(code_response)

        elif code_response and code_response != "" and ("js" in code_response or "javascript" in code_response):
            lang = "javascript"
            exec_out, code = self.javascript_sandbox(code_response)

        elif code_response and code_response != "" and "voltscript" in code_response:
            lang = "voltscript"
            exec_out, code = self.voltscript_sandbox(code_response)

        print(exec_out)
        print(code)
        return exec_out, code, lang, code_response


class CodeAssistantPipeline(BasePipeline):
    def __init__(self, docs: List[Document], model: str = "gemini"):
        """
        Initialize LLM, Code Interpreter's prompt template, vector store, and retriever.
        """
        # models = ['codey', 'gemini', 'palm']
        # assert model in models, f"model should only be {', '.join(models)}"
        self.output_schema = CodeInterpreterSchema()
        prompt_template = CodeAssistantPromptTemplate(self.output_schema.response_format).prompt
        super().__init__(docs, model, prompt_template)

    def parse_llm_response(self, llm_response: str) -> Dict:
        try:
            parsed_output = self.output_schema.output_parser.parse(llm_response)
        except Exception as e:
            try:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", llm_response)
                parsed_output = self.output_schema.output_parser.parse(parsed_response_palm)
            except Exception as e:
                try:
                    response_dic = llm_response[llm_response.find("{"):llm_response.find("}") + 1]
                    parsed_output = eval(response_dic.replace('null', '""'))
                except Exception as e:
                    parsed_output = {'bot_response': llm_response, 'code': ''}

        return parsed_output

    def __call__(self, query: str, model: str = None) -> Tuple[Dict, str, List[str]]:
        """
        Invoked for RAG response. For non-RAG response, invoke LLM directly.
        """
        response = self.chain({"query": query})
        source_docs = [x.page_content for x in response['source_documents']]
        llm_response = self.parse_llm_response(response["result"])
        exec_out, code, lang, code_response = self.run_code(llm_response.get('code'))

        return llm_response, exec_out, source_docs


class CodeInterpreterPipeline(BasePipeline):
    def __init__(self, docs: List[Document], model: str = "codey"):
        """
        Initialize LLM, Code Interpreter's prompt template, vector store, and retriever.
        """
        # models = ['codey']
        # assert model in models, f"model should only be {', '.join(models)}"
        prompt_template = CodeInterpreterPromptTemplate().prompt
        super().__init__(docs, model, prompt_template)

    def __call__(self, query: str, model: str = None) -> Tuple[str, str, List[str]]:
        """
        Invoked for RAG response. For non-RAG response, invoke LLM directly.
        """
        response = self.chain({"query": query})
        source_docs = [x.page_content for x in response['source_documents']]
        llm_response = response["result"]
        exec_out, code, lang, code_response = self.run_code(llm_response)

        return llm_response, exec_out, source_docs
