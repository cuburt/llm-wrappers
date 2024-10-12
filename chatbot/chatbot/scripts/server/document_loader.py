# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
import os
from typing import List
import numpy as np
import pandas as pd
from langchain.schema.document import Document
from scripts.log import logger


def txt_to_list_of_str(dir: str="") -> List[str]:
    try:
        list_of_texts = []
        files = [f for f in os.listdir(dir) if f.endswith(".txt")]
        for file in files:
            with open(file, "r", encoding="utf-8") as ifile:
                texts = []
                for line in ifile:
                    texts.append(line)
            list_of_texts.append(" ".join(texts))
    except Exception as e:
        logger.error(str(e))

    return list_of_texts


def txt_to_list_of_docs(dir: str="") -> List[Document]:
    try:
        list_of_docs = []
        files = [f for f in os.listdir(dir) if f.endswith(".txt")]
        for file in files:
            with open(os.path.join(dir, file), "r", encoding="utf-8") as ifile:
                texts = []
                for line in ifile:
                    texts.append(line)
            doc = Document(page_content=" ".join(texts))
            list_of_docs.append(doc)
    except Exception as e:
        logger.error(str(e))

    return list_of_docs


def crawled_csv_to_list_of_str(dir: str="") -> List[str]:
    try:
        list_of_texts = []
        files = [f for f in os.listdir(dir) if f.endswith(".csv")]
        for i, df in enumerate([pd.read_csv(os.path.join(dir, csv)) for csv in files]):
            for column in df.columns:
                for r in df[column].unique():
                    if r is not None or r != pd.NA or r!=np.nan:
                        list_of_texts.append(r)
    except Exception as e:
        logger.error(str(e))

    return list_of_texts


def crawled_csv_to_list_of_docs(dir: str="") -> List[Document]:
    try:
        list_of_docs = []
        files = [f for f in os.listdir(dir) if f.endswith(".csv")]
        for i, df in enumerate([pd.read_csv(os.path.join(dir, csv)) for csv in files]):
            for column in df.columns:
                for r in df[column].unique():
                    if r is not None or r != pd.NA or r != np.nan:
                        doc = Document(page_content=r)
                        list_of_docs.append(doc)
    except Exception as e:
        logger.error(str(e))

    return list_of_docs


def column_in_csv_to_list_of_docs(col:str, metadata_cols: List[str], file:str) -> List[Document]:
    try:
        df = pd.read_csv(file)
        docs = []
        for i in range(0, len(df[col].values)):
            doc = Document(page_content=df[col].values[i], metadata={col: df[col].values[i] for col in metadata_cols})
            docs.append(doc)
    except Exception as e:
        logger.error(str(e))

    return docs
