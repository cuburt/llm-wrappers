# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
from typing import List, Any, Tuple, Dict
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever, Document
from scripts.server.encoders import (MsMarcoMiniLML6V2,
                                     BAAILLMEmbedder,
                                     BAAIBiEncoder,
                                     BAAICrossEncoder,
                                     Stella400MV5,
                                     Stella1_5BV5,
                                     NVEmbedV1,
                                     SFREmbedding2R,
                                     SFREmbeddingMistral)
from langchain_community.document_transformers import LongContextReorder
from scripts.log import logger
import time
from pathlib import Path
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent)


def sort_documents_by_score(documents: List[Document], scores: List[Tuple[float, str]]) -> List[Document]:
    # Create a dictionary for quick lookup of scores by page_content
    score_dict = {content: score for score, content in scores}

    # Sort documents based on the scores using the score dictionary
    sorted_documents = sorted(documents, key=lambda doc: score_dict.get(doc.page_content, float('inf')))

    return sorted_documents


class Rerouter:
    def __init__(self):
        self.short_term_memory: List = []
        self.route = "voltmx-community"

    def to_reroute(self) -> bool:
        try:
            # Clear memory, leave most recent query and the current query
            self.short_term_memory = self.short_term_memory[-2:]
            # Return if they are the same
            return self.short_term_memory[0] == self.short_term_memory[1]
        except IndexError:
            logger.info("No previous query. No reroute options...")
            return False


class RetrieverFactory(BaseRetriever):
    rerouter: Any = Rerouter()
    vectorstores: Dict
    router: Any = None
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    k: int = 5
    skip_longcontext_reorder: bool = False
    cross_encoder = BAAILLMEmbedder()

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        reranked_docs = []

        start = time.time()
        self.rerouter.short_term_memory.append(query.strip())
        route = self.rerouter.route
        if not self.rerouter.to_reroute():
            route = self.router(query)
        logger.info(f"Routed to {route} chain.")
        logger.info(f"Router RT: {(time.time() - start):.4f}")
        print(self.rerouter.short_term_memory)

        if not route:
            # route can be replaced by None self.rerouter.to_reroute() == False
            return []

        try:
            start = time.time()
            # relevant_docs = self.vectorstores[route].vectorstore.search(query, search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8, "k": self.k})
            # relevant_docs = self.vectorstores[route].vectorstore.search(query, search_type="similarity", search_kwargs={"k": self.k})
            relevant_docs = self.vectorstores[route].vectorstore.search(query, search_type=self.search_type, k=self.k)
            logger.info(f"Bi-Encoder RT: {(time.time() - start):.4f}")
            page_contents = [doc.page_content for doc in relevant_docs]
        except Exception as e:
            logger.error(str(e))
            relevant_docs, page_contents = [], []

        try:
            start = time.time()
            pairs = []
            for content in page_contents:
                pairs.append([query, content])
            scores = self.cross_encoder.predict(pairs)
            scored_docs = zip(scores, page_contents)
            sorted_contents = sorted(scored_docs, reverse=True)
            reranked_docs = sort_documents_by_score(relevant_docs, sorted_contents)
            logger.info(f"Cross-Encoder RT: {(time.time() - start):.4f}")
            relevant_docs = reranked_docs
        except Exception as e:
            logger.error(str(e))

        try:
            if not self.skip_longcontext_reorder:
                reordering = LongContextReorder()
                start = time.time()
                relevant_docs = reordering.transform_documents(relevant_docs)
                logger.info(f"LC-Reranker RT: {(time.time() - start):.4f}")
        except Exception as e:
            logger.error(str(e))

        return list(relevant_docs if not reranked_docs else reranked_docs)

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()
