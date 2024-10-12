from langchain_core.retrievers import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import Any, List


class Retriever(BaseRetriever):
    vectorstore: Any
    image_paths: List[str] = None
    k: int

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # query preprocess
        q, retrieved_images, indices, distances = self.vectorstore.retrieve_similar_images(
            query,  # our search query
            k=self.k,  # return 3 most relevant docs
            image_paths=self.image_paths
        )

        return [self.vectorstore.get_doc_image_from_index(i) for i in indices[0]]