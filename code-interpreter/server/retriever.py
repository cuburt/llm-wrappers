from typing import List, Any, Dict
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain.schema.retriever import BaseRetriever, Document
from langchain.vectorstores import VectorStore
from sentence_transformers import CrossEncoder
from langchain.document_transformers import LongContextReorder


class RetrieverFactory(BaseRetriever):
    vectorstore: VectorStore
    """VectorStore to use for retrieval."""
    search_type: str = "similarity"
    """Type of search to perform. Defaults to "similarity"."""
    k: int = 4
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vectorstore_docs = self.vectorstore.search(query, self.search_type, k=10)
        # page_contents = [doc.page_content for doc in vectorstore_docs]
        pairs = []
        for doc in vectorstore_docs:
            pairs.append([query, doc.page_content])
        scores = self.cross_encoder.predict(pairs)
        scored_docs = zip(scores, vectorstore_docs)
        sorted_docs = sorted(scored_docs, reverse=True)
        reranked_docs = [doc for _, doc in sorted_docs][0:self.k]
        reordering = LongContextReorder()
        result_docs = reordering.transform_documents(reranked_docs)

        return list(result_docs)

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()
