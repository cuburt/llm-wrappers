# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon,manjunath_hegde

@project: XAI

@input:

@output:

@des
"""
import uuid
import time
from typing import List, Dict, Tuple, Any, Sequence
from scripts.log import logger
from scripts.server.llm import PalmLLM, GeminiLLM, CodeyLLM, Gemini15LLM, GeminiFlashLLM, Llama3LLM, GPT4oLLM, GPT4oMiniLLM
from scripts.server.output_parser import CodeInterpreterSchema
from scripts.server.vector_index import FAISSStore, PineconeStore
from scripts.server.prompt_template import IQPromptTemplate, MemoryPromptTemplate, DocumentParserPromptTemplate
from scripts.server.retriever import RetrieverFactory
from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from pathlib import Path, PurePath
from scripts.data_preprocess.data_pipeline import DataPipeline
from scripts.server.session import Session
from scripts.server.routers import SemanticRouter


class ConversationBufferWindowMemory(InMemoryChatMessageHistory):
    """
    A chat message history that only keeps the last K messages.
    """

    buffer_size: int

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if self.buffer_size > 0:
            self.messages.extend(messages)
            self.messages = self.messages[-(self.buffer_size * 2):]
        else:
            self.messages = []


class Chain:
    def __init__(self, data_source: Dict, vectorstore_config: Dict, memory_window: int):
        """
        Initialize LLM, Code Interpreter's prompt template, vector store, and retriever.
        """
        try:
            self.palm = PalmLLM()
            self.gemini = GeminiFlashLLM()
            self.llama = Llama3LLM()
            self.gpt = GPT4oMiniLLM()
            self.session_manager = Session()
            self.memory_window = memory_window
            self.chat_history = {}
            self.default_session_id = str(uuid.uuid4())
            vectorstores = self.init_vectorstore(data_source=data_source,
                                                 db_dir=vectorstore_config.get('faiss_dir'),
                                                 embed_model=vectorstore_config.get('embed_model'),
                                                 force_load=vectorstore_config.get("force_load"),
                                                 max_filesize_bytes=vectorstore_config.get("max_filesize_bytes"),
                                                 insert_batch_size=vectorstore_config.get("insert_batch_size"))

            retriever = RetrieverFactory(vectorstores=vectorstores,
                                          k=vectorstore_config.get('top_k'),
                                          skip_longcontext_reorder=vectorstore_config.get('skip_longcontext_reorder'),
                                          search_type=vectorstore_config.get('search_type'))
            retriever.router = SemanticRouter()
            self.palm_custom_chain = self.build_qa_chain(self.palm, IQPromptTemplate().prompt, retriever)
            self.gemini_custom_chain = self.build_qa_chain(self.gemini, IQPromptTemplate().prompt, retriever)
            self.gpt_custom_chain = self.build_qa_chain(self.gpt, IQPromptTemplate().prompt, retriever)
            self.llama_custom_chain = self.build_qa_chain(self.llama, IQPromptTemplate().prompt, retriever)

        except Exception as e:
            raise Exception(str(e))

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves history from BigQuery
        """
        if session_id not in self.chat_history:
            history = []
            try:
                start = time.time()
                history = self.session_manager.get_conversations(session_id, self.memory_window)
                logger.info(f"BQSelect RT: {(time.time() - start):.4f}")
            except Exception as e:
                logger.warning(str(e))
            self.chat_history[session_id] = ConversationBufferWindowMemory(buffer_size=self.memory_window)
            for entry in reversed(list(history)):
                self.chat_history[session_id].add_user_message(entry["message"])
                self.chat_history[session_id].add_ai_message(entry["response"])
        return self.chat_history[session_id]

    def build_qa_chain(self, llm, prompt_template, retriever) -> ConversationalRetrievalChain:
        """
        QA chain's constructor
        """
        # question_generator = create_history_aware_retriever(
        #     llm, retriever, MemoryPromptTemplate().prompt
        # )
        doc_chain = create_stuff_documents_chain(llm,
                                                 prompt=prompt_template,
                                                 document_prompt=DocumentParserPromptTemplate().prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        return RunnableWithMessageHistory(rag_chain,
                                          get_session_history=self.get_session_history,
                                          input_messages_key="input",
                                          history_messages_key="chat_history",
                                          output_messages_key="answer")

    @staticmethod
    def run_qa_chain(qa_chain, query: str, session_id: str):
        """
        Retriever's function. can be used in function calling or as agent tool
        """
        try:
            return qa_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
        except Exception as e:
            raise Exception(str(e))

    @staticmethod
    def init_vectorstore(data_source:Dict, db_dir:str, embed_model:str, force_load:bool=False, max_filesize_bytes:float=0.0, insert_batch_size:int=100) -> Dict:
        """
        Vectorstore's constructor. Builds vectorstores based on data_path and vectorstore configuration in config.json
        output: {"voltmx": FAISS()}
        """
        try:
            logger.info("Initializing vectorstores...")
            vectorstores = {}
            data_loader = DataPipeline(data_source)
            for data_path in data_source['local']['data_path']:
                _key = PurePath(data_path).name
                if not force_load:
                    try:
                        data = data_loader.run(data_path)
                        vectorstores.update({_key: FAISSStore(docs={"filepath": data_path, "documents": data},
                                                              db_dir=db_dir,
                                                              db_file=_key,
                                                              model=embed_model,
                                                              max_filesize_bytes=max_filesize_bytes,
                                                              insert_batch_size=insert_batch_size)})
                    except AssertionError as ae:
                        logger.error(str(ae))
                else:
                    vectorstores.update({_key: FAISSStore(docs={"filepath": None, "documents": None},
                                                          db_dir=db_dir,
                                                          db_file=_key,
                                                          model=embed_model)})

        except Exception as e:
            raise Exception(str(e))

        return vectorstores

    @staticmethod
    def format_query(query: str) -> str:
        """
        Formats query with a template instruction.
        """
        formatted_prompt = ''
        try:
            formatted_prompt = CodeInterpreterSchema().prompt.format(query=query)
        except Exception as e:
            logger.error(str(e))

        return formatted_prompt

    def __call__(self, query: str, llm: str = 'gemini', task: str = 'docs', enable_rag=True, session_id: str = None) -> Tuple[
        str, str, List[Tuple[Any, Any]]]:
        """
        Invoked for RAG response. For non-RAG response, invoke LLM directly.
        """
        try:
            if not enable_rag:
                response, source_docs = {"answer": eval(f"self.experimental_chain.{llm}")}, []
            else:
                response = self.run_qa_chain(
                    qa_chain=eval(f"self.{llm}_custom_chain"),
                    query=query,
                    session_id=session_id if session_id else self.default_session_id)
                source_docs = [(x.page_content, x.metadata['source']) for x in response['context']]
            assert response["answer"], "'answer' key not found in model response."
            message_id = str(uuid.uuid4())
            if session_id:
                start = time.time()
                self.session_manager.add_to_conversations(message_id=message_id, session_id=session_id, message=str(query), response=str(response["answer"]))
                logger.info(f"BQUpdate RT: {(time.time() - start):.4f}")
            return message_id, response["answer"].encode("utf-8"), source_docs
        except Exception as e:
            logger.error(str(e))
            raise Exception(str(e))
