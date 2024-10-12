# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
from typing import List
from scripts.server.encoders import (SFREmbedding2R,
                                     SFREmbeddingMistral,
                                     Stella400MV5,
                                     Stella1_5BV5,
                                     BAAIBiEncoder,
                                     BAAILLMEmbedder,
                                     NVEmbedV1,
                                     AllMiniLML6V2,
                                     TextEmbeddingGecko,
                                     MultilingualEmbeddingGecko,
                                     TextMultimodalEmbeddingGecko)
from langchain.schema.embeddings import Embeddings
from torch import Tensor
from scripts.log import logger
from pathlib import Path
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent) #set project root directory


class TextEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = eval(f"{model_name}()")

    def embed_query(self, text: str) -> List[float]:
        try:
            if self.model_name in ['BAAILLMEmbedder', 'BAAIBiEncoder', 'NVEmbedV1']:
                embeddings = self.model.encode(text, normalize_embeddings=True)
            else:
                embeddings = self.model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(str(e))

    def embed_documents(self, texts: List[str]) -> List[List[Tensor]]:
        try:
            if self.model_name in ['BAAILLMEmbedder', 'BAAIBiEncoder', 'NVEmbedV1']:
                embeddings = [self.model.encode(text, normalize_embeddings=True) for text in texts]
            else:
                embeddings = [self.model.encode(text) for text in texts]
            return embeddings
        except Exception as e:
            logger.error(str(e))
