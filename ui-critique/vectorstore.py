from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import os
import base64
import io
import re
from langchain.docstore.document import Document
from PIL import Image
import torch
from data import DataLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_experimental.open_clip import OpenCLIPEmbeddings

class ImageFAISS(FAISS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _encode_image_to_base64(image_path: str=None, _bytes=None) -> str:
        """Helper function to convert an image to base64 string."""
        base64_string = ""
        if image_path:
            with open(image_path, "rb") as image_file:
                base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        elif _bytes:
            base64_string = base64.b64encode(_bytes).decode('utf-8')
        assert base64_string, "No image was encoded."
        return base64_string

    @staticmethod
    def _decode_base64_to_image(base64_string: str) -> Image.Image:
        """Helper function to decode base64 string back to image."""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    @staticmethod
    def get_file_extension_from_filepath(filepath: str) -> str:
        # Split the filepath and get the extension
        _, file_extension = os.path.splitext(filepath)
        return file_extension.strip(".").lower()

    def image_features_from_bytes(self, bytes):
        embeddings = []
        for byte in bytes:
            image = Image.open(io.BytesIO(byte))
            inputs = self.embedding_function.preprocess(image).unsqueeze(0)
            embeddings_tensor = self.embedding_function.model.encode_image(inputs)
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            embeddings.append(embeddings_list)
        return embeddings

    def image_features_from_images(self, images):
        embeddings = []
        for image in images:
            inputs = self.embedding_function.preprocess(image).unsqueeze(0)
            embeddings_tensor = self.embedding_function.model.encode_image(inputs)
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            embeddings.append(embeddings_list)
        return embeddings

    def from_images(self, image_uris: list[str]=None, image_bytes=None) -> FAISS:
        if image_uris:
            docs_to_add = {}
            for i, image_uri in enumerate(image_uris):
                base64_image = self._encode_image_to_base64(image_path=image_uri)
                document = Document(page_content=f"data:image/{self.get_file_extension_from_filepath(image_uri)};base64,{base64_image}", metadata={"base64_image": base64_image})
                doc_id = str(i)
                docs_to_add[doc_id] = document
                self.index_to_docstore_id[i] = doc_id
            self.docstore.add(docs_to_add)

            image_features = np.array(self.embedding_function.embed_image(image_uris))

        else:
            docs_to_add = {}
            for i, _bytes in enumerate(image_bytes):
                base64_image = self._encode_image_to_base64(_bytes=_bytes)
                document = Document(page_content=f"data:image/jpeg;base64,{base64_image}", metadata={"base64_image": base64_image})
                doc_id = str(i)
                docs_to_add[doc_id] = document
                self.index_to_docstore_id[i] = doc_id
            self.docstore.add(docs_to_add)

            image_features = np.array(self.image_features_from_bytes(image_bytes))

        self.index = faiss.IndexFlatIP(image_features.shape[1])
        self.index.add(image_features)

        return self

    def get_doc_image_from_index(self, idx: int) -> Document:
        """Retrieve base64 encoded image from the docstore and convert back to an image."""
        doc_id = self.index_to_docstore_id[idx]
        document = self.docstore.search(doc_id)
        return document

    def retrieve_similar_images(self, query, image_paths=None, k=3):
        if type(query) != str:
            query = query["image"] if query["image"] else query["question"]
        assert type(query) == str, "Sorry an error was encountered while processing the query."
        base64_pattern = r'(data:image\/(?:jpeg|png);base64,[A-Za-z0-9+/=]+)'
        matches = re.findall(base64_pattern, query)
        if matches:
            images = [self._decode_base64_to_image(_match.replace("data:image/jpeg;base64,", '').replace("data:image/png;base64,", '')) for _match in matches]
            query_features = np.array(self.image_features_from_images(images))
        else:
            query_features = np.array(self.embedding_function.embed_query(query))

        query_features = query_features.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_features, k)
        retrieved_images = []

        if image_paths:
            retrieved_images = [image_paths[int(idx)] for idx in indices[0]]

        return query, retrieved_images, indices, distances


class Vectorstore:
    def __init__(self):
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        clip_embd = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k", device=device)
        images = DataLoader(rows=5).images
        faiss_obj = ImageFAISS(embedding_function=clip_embd,
                               index=None,
                               docstore=InMemoryDocstore(),
                               index_to_docstore_id={},
                               normalize_L2=False,
                               distance_strategy=DistanceStrategy.COSINE,
                               # EUCLIDEAN_DISTANCE, MAX_INNER_PRODUCT, DOT_PRODUCT, JACCARD, COSINE
                               )
        vectorstore_dir = "faiss"
        if not os.path.exists(vectorstore_dir):
            self.vectorstore = faiss_obj.from_images(image_bytes=images)
            self.vectorstore.save_local(vectorstore_dir)
        try:
            self.vectorstore = faiss_obj.load_local(folder_path=vectorstore_dir, embeddings=clip_embd)
        except:
            self.vectorstore = faiss_obj.load_local(folder_path=vectorstore_dir, embeddings=clip_embd,
                                               allow_dangerous_deserialization=True)