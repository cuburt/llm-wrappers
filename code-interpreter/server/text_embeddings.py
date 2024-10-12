from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
from langchain.schema.embeddings import Embeddings


class TextEmbeddings(Embeddings):

    def __init__(self, model_name):

        self.model_name: str = model_name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.model.encode(text)
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = [self.model.encode(text) for text in texts]
        return embeddings
