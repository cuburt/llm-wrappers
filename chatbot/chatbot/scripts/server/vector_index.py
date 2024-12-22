# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon, somya.upadhyay

@project: XAI

@input:

@output:

@des
"""
import os
from typing import List, Dict
from langchain.schema.vectorstore import VectorStore
import pinecone
from langchain_community.vectorstores import Pinecone, FAISS
from scripts.log import logger
from scripts.server.adapter import ArxivStoreAdapter
from scripts.server.text_embeddings import TextEmbeddings
from langchain.schema.document import Document
from pathlib import Path
import faiss
import threading
import platform


def get_most_recently_updated_file(folders):
    most_recent_file = None
    most_recent_time = None

    for folder in folders:
        # Get a list of all files in the folder with their full paths
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        for file in files:
            file_mtime = os.path.getmtime(file)
            if most_recent_time is None or file_mtime > most_recent_time:
                most_recent_time = file_mtime
                most_recent_file = file

    return Path(most_recent_file).parent


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)
def is_file_too_big(file_path, max):
    if platform.system() == 'Windows':
        dir_path = file_path.split('/')
        dir_path = os.path.join(PROJECT_ROOT_DIR, *dir_path)
    else:
        dir_path = os.path.join(PROJECT_ROOT_DIR, file_path)
    size = sum([os.path.getsize(file) for file in os.scandir(dir_path)])
    logger.info(f"{file_path}'s size in byte is {size}")
    return size >= max


class FAISSStore:
    def __init__(self, docs: Dict[str, List[Document]]=None, db_dir: str=None, db_file: str=None, model: str="BAAILLMEmbedder", max_filesize_bytes:float=0.0, insert_batch_size:int=100):
        logger.info(f"Initialising {db_file} vectorstore...")
        # logger.info(f"GPU: {faiss.get_num_gpus()}")
        vectorstore_dir = os.path.join(str(Path(__file__).parent.parent.parent), db_dir)
        if not os.path.exists(vectorstore_dir):
            os.mkdir(vectorstore_dir)

        self.local_dir = os.path.join(str(Path(__file__).parent.parent.parent), db_dir, f"{db_file}-{model}")
        self.vectorstore: FAISS
        if not docs["documents"] and db_dir and db_file:
            try:
                if not os.path.exists(self.local_dir):
                    # vectorstore directory
                    vectorestore_folder = os.path.join(str(Path(__file__).parent.parent.parent), db_dir)
                    candidate_vectorstores = [os.path.join(vectorestore_folder, dir) for dir in os.listdir(vectorestore_folder) if db_file in dir]
                    assert candidate_vectorstores, f"No existing vectorstore for {db_file}."
                    self.local_dir = get_most_recently_updated_file(candidate_vectorstores)
                    assert self.local_dir, f"Corrupt vectorstore for {db_file}."
                    self.embed_model = TextEmbeddings(os.path.basename(os.path.normpath(self.local_dir)).strip(f"{db_file}-"))
                else:
                    self.embed_model = TextEmbeddings(model)
            except AssertionError as e:
                logger.error(str(e))

            try:
                self.vectorstore = FAISS.load_local(folder_path=self.local_dir, embeddings=self.embed_model, allow_dangerous_deserialization=True)
            except Exception as e:
                logger.warning(str(e))
                self.vectorstore = FAISS.load_local(folder_path=self.local_dir, embeddings=self.embed_model)
            logger.info("Vectorstore successfully loaded.")

        elif docs["documents"] and not is_file_too_big(docs["filepath"], max_filesize_bytes):
            self.embed_model = TextEmbeddings(model)
            self.vectorstore = FAISS.from_documents(docs["documents"], self.embed_model)
            if not os.path.exists(self.local_dir):
                os.mkdir(self.local_dir)
            self.vectorstore.save_local(folder_path=self.local_dir)
            logger.info(f"VECTORSTORE saved at: {self.local_dir}\nFileSize: {convert_bytes(sum([os.path.getsize(file) for file in os.scandir(self.local_dir)]))}")

        elif docs["documents"] and is_file_too_big(docs["filepath"], max_filesize_bytes):
            self.embed_model = TextEmbeddings(model)
            logger.info("Asynchronous vectorstore builder initialised.")
            self.vectorstore = FAISS.from_documents(docs["documents"][:insert_batch_size], self.embed_model)
            if not os.path.exists(self.local_dir):
                os.mkdir(self.local_dir)
            self.vectorstore.save_local(folder_path=self.local_dir)
            logger.info(f"Initial vectorstore saved at: {self.local_dir}\nFileSize: {convert_bytes(sum([os.path.getsize(file) for file in os.scandir(self.local_dir)]))}\nAsynchronous update begins...")
            # Allows parallel inference and vectorstore build.
            # TODO: do multithreading for GPU setup.
            insert_docs_thread = threading.Thread(target=self.insert_docs, daemon=True, name="insert_docs", args=[docs["documents"][insert_batch_size:], insert_batch_size])
            insert_docs_thread.start()

    def insert_dict(self, texts: List[Dict], metadata_cols=None):
        docs = []
        try:
            for i in range(0, len(texts)):
                doc = Document(page_content=texts[i]['text'], metadata= {m: texts[i].get(m, "/") if m in texts[i] else "/" for m in metadata_cols})
                docs.append(doc)
            res = self.vectorstore.add_documents(docs)
            if self.local_dir:
                self.vectorstore.save_local(str(self.local_dir))
        except Exception as e:
            logger.error(str(e))
            res = str(e)

        return res

    def insert_docs(self, docs: List[Document], insert_batch_size):
        try:
            # Process the list in batches of 5
            for i in range(0, len(docs), insert_batch_size):
                batch = docs[i:i + insert_batch_size]
                logger.info(f"Embedding {len(batch)} documents...")
                # Process each batch (replace this comment with your processing logic)
                res = self.vectorstore.add_documents(batch)
                logger.info(f"Documents embeddings for: {res}")
                if self.local_dir:
                    self.vectorstore.save_local(str(self.local_dir))
                    logger.info(
                        f"VECTORSTORE updated at: {self.local_dir}\nFileSize: {convert_bytes(sum([os.path.getsize(file) for file in os.scandir(self.local_dir)]))}")
        except Exception as e:
            logger.error(str(e))


class PineconeStore:
    def __init__(self, index_name, api_key, model: str="AllMiniLML6V2"):
        try:
            logger.info("initialising vector database...")
            pinecone.init(api_key= api_key, environment="gcp-starter")
            index = pinecone.Index(index_name)
            embed_model = TextEmbeddings(model)
            label = 'text'
            # vectorstore
            self.vectorstore = Pinecone(
                index, embed_model.embed_query, label)
        except Exception as e:
            logger.error(str(e))


class ArxivStore(VectorStore):

    def _post(self, payload, headers):
        try:
            result = ArxivStoreAdapter().post(payload=payload, headers=headers, endpoint=":predict")
        except Exception as e:
            logger.error(str(e))       
        return result

    def _get(self, payload, headers):
        try:
            result = ArxivStoreAdapter().get(payload=payload, headers=headers)
        except Exception as e:
            logger.error(str(e))
        return result

    def similarity_search(self, query) -> List[Document]:
        try:
            payload = ArxivStoreAdapter().build_payload(query)
            headers = ArxivStoreAdapter().build_headers()
            docs = []
            results = self._post(payload, headers)
            for res in results["predictions"]:
                # metadata = res["metadata"]
                # if self._text_key in metadata:
                #     text = metadata.pop(self._text_key)
                #     score = res["score"]
                score=res[:6]
                score = float(score.replace(': ',''))
                text=res[6:]
                docs.append(Document(page_content=str(text)))
        except Exception as e:
            logger.error(str(e))

        return docs

    def add_texts(self):
        return True

    def from_texts(self):
        return True
