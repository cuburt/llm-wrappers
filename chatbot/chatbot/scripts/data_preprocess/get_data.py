# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: somya.upadhyay, cuburt.balanon,manjunath_hegde

@project: XAI

@input:

@output:

@des
"""
from typing import List, Iterable
import pandas as pd
import glob
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader,UnstructuredHTMLLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from scripts.log import logger
from google.cloud import bigquery
from google.cloud import storage
from pathlib import Path
import platform
from langchain.text_splitter import NLTKTextSplitter
import nltk


PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)


class GetData:

    def __init__(self, data_source_config):
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.data_types = data_source_config["data_types"]
        self.data_type_supported = data_source_config["data_type_supported"]
        self.chunk_size = data_source_config["chunk_size"]
        self.chunk_overlap = data_source_config["chunk_overlap"]
        self.allow_chunking = data_source_config["allow_chunking"]
        self.cloud_config = data_source_config["cloud"]

        if data_source_config['data_warehouse'] == 'cloud':
            self.cloud_storage_loader()

        if self.data_types == ["all"]:
            self.data_types = ["csv", "pdf", "html", "txt"]
    
    def chunk_data(self, docs: Iterable[Document]) -> List[Document]:
        text_splitter = NLTKTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap,length_function=len,is_separator_regex=False)
        docs = text_splitter.split_documents(docs)
        return docs
    
    def cloud_storage_loader(self):
        try:
            # Initialise a client
            storage_client = storage.Client( self.cloud_config["project"])
            # Create a bucket object for our bucket
            bucket = storage_client.get_bucket( self.cloud_config["bucket"])
            # Create a blob object from the filepath
            blobs = bucket.list_blobs(prefix= self.cloud_config["path"])
            # Download the files to a destination
            directory_path=self.cloud_config["local_data_path"]
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            else:
                shutil.rmtree(directory_path, ignore_errors=True)
                os.makedirs(directory_path)
            for blob in blobs:
                local_file_path = os.path.join(directory_path, os.path.basename(blob.name))
                blob.download_to_filename(local_file_path)
        except Exception as e:
            logger.error(str(e))

    @staticmethod
    def collect_files(postfix, dir_path):
        try:
            if platform.system() == 'Windows':
                dir_path = dir_path.split('/')
                dir_path = os.path.join(PROJECT_ROOT_DIR, *dir_path)
            else:
                dir_path = os.path.join(PROJECT_ROOT_DIR, dir_path)

            assert os.path.lexists(dir_path), f"Data of type {postfix} does not exists"
            
            all_files = glob.glob(f'{dir_path }/**/*.*{postfix}', recursive=True)
            all_files.extend(glob.glob(f'{dir_path }/*.{postfix}', recursive=True))

            assert all_files, f"No files found in {dir_path}."

            return all_files

        except Exception as e:
            # handle error in the next except block
            raise Exception(str(e))

    def read_csv_files(self, dir_path):
        try:
            all_files = self.collect_files("csv", dir_path)
            text_data = []
            for filename in all_files:
                html = pd.read_csv(filename, encoding='utf-8')
                html.dropna(inplace=True)
                for i in range(len(html)-1):
                    try:
                        page_contents = html.iloc[[i]]['page_content'].values
                        sources = html.iloc[[i]]['source'].values
                        assert len(page_contents) == 1, "Document may not contain multiple 'page_content'"
                        assert len(sources) == 1, "Document may not contain multiple 'source'"
                        text_data.append(Document(page_content=page_contents[0], metadata={"source": sources[0]}))
                    except Exception as e:
                        logger.warning(str(e))

            return text_data
        except Exception as e:
            # handle error in the next except block
            raise Exception(str(e))

    def read_html_files(self, dir_path, data_type):
        try:
            all_files = self.collect_files(data_type, dir_path)
            text_data = []
            for filename in all_files:
                try:
                    loader = UnstructuredHTMLLoader(filename)
                    loaderpages = loader.load()
                    text_data.extend(loaderpages)
                except Exception as e:
                    logger.warning(str(e))

            return text_data
        except Exception as e:
            # handle error in the next except block
            raise Exception(str(e))

    def read_pdf_files(self, dir_path):
        try:
            all_files = self.collect_files("pdf", dir_path)
            text_data = []
            for filename in all_files:
                try:
                    loader = PyPDFLoader(filename)
                    loaderpages = loader.load_and_split()
                    text_data.extend(loaderpages)
                except Exception as e:
                    logger.warning(str(e))

            return text_data
        except Exception as e:
            # handle error in the next except block
            raise Exception(str(e))

    def read_other_files(self, dir_path, data_type):
        try:
            all_files = self.collect_files(data_type, dir_path)
            text_data = []
            for filename in all_files:
                try:
                    loader = TextLoader(filename)
                    loaderpages=loader.load_and_split()
                    text_data.extend(loaderpages)
                except Exception as e:
                    logger.warning(str(e))

            return text_data
        except Exception as e:
            # handle error in the next except block
            raise Exception(str(e))

    def __call__(self, data_path):
        data_set = []
        for data_type in self.data_types:
            try:
                if data_type == 'csv':
                    data = self.read_csv_files(data_path)
                elif data_type == 'html' or data_type == 'txt':
                    data = self.read_html_files(data_path, data_type)
                elif data_type == 'pdf':
                    data = self.read_pdf_files(data_path)
                else:
                    data = self.read_other_files(data_path, data_type)
                data_set.extend(data)
            except Exception as e:
                # non-breaking exception
                logger.warning(str(e))

        # throw error if directory is completely empty or incorrect. handle in the next except block.
        assert data_set, f"No data Found in {data_path}"
        if self.allow_chunking:
            data_set = self.chunk_data(data_set)

        return data_set
