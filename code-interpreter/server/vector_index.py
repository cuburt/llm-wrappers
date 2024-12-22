from typing import List, Dict
import logging
from text_embeddings import TextEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.document import Document


class VectorStoreFactory:
    def __init__(self, docs: List[Document]=None, db_dir: FAISS=None):
        logging.info("initialising vector database...")
        embed_model = TextEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
        self.local_dir = db_dir
        if not docs and self.local_dir:
            self.vectorstore = FAISS.load_local(self.local_dir, embed_model)
        elif docs and not db_dir:
            self.vectorstore = FAISS.from_documents(docs, embed_model)
        else:
            raise Exception("No Documents and local FAISS")

    def insert_docs(self, texts: List[Dict], metadata_cols: List = None):
        if metadata_cols is None:
            metadata_cols = []
        docs = []
        for i in range(0, len(texts)):
            doc = Document(page_content=texts[i]['text'], metadata= {m: texts[i].get(m, "/") if m in texts[i] else "/" for m in metadata_cols if metadata_cols})
            docs.append(doc)
        res = self.vectorstore.add_documents(docs)
        if self.local_dir:
            self.vectorstore.save_local(self.local_dir)
        return res
