# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
from typing import Dict, List
import numpy as np
import json
import subprocess
import platform
import requests
from pathlib import Path
import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from scripts.log import logger
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent) #set project root directory


class EmbedRequest:
    def __call__(self, model_id:str, data:Dict):
        try:
            region = "us-central1"
            api_endpoint = f"{region}-aiplatform.googleapis.com"
            project_id = "hclsw-gcp-xai"
            assert model_id in ["textembedding-gecko@003", "textembedding-gecko-multilingual@001", "multimodalembedding@001", "text-embedding-004"], f"Sorry {model_id} is not currently supported."
            script_file_path = os.path.join(PROJECT_ROOT_DIR, "scripts/server", "access_token.sh")
            cmd = ['/bin/bash', script_file_path]
            if platform.system() == 'Windows':
                cmd = ['C:/Program Files/Git/bin/bash.exe', script_file_path]
            access_token = subprocess.run(cmd, capture_output=True, text=True)
            assert access_token.stdout, access_token.stderr
            headers = {"Authorization": f"Bearer {access_token.stdout}", "Content-Type": "application/json"}
            url = f"https://{api_endpoint}/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_id}:predict"
            return requests.post(url=url, headers=headers, data=json.dumps(data)).json()
        except Exception as e:
            logger.error(str(e))


class MsMarcoMiniLML6V2(CrossEncoder):
    def __init__(self):
        super().__init__('cross-encoder/ms-marco-MiniLM-L-6-v2')


class AllMiniLML6V2(SentenceTransformer):
    def __init__(self):
        super().__init__('sentence-transformers/all-MiniLM-L6-v2')


class NVEmbedV1(SentenceTransformer):
    def __init__(self):
        api_key_path = "scripts/server/.hf_read.txt"
        if platform.system() == 'Windows':
            api_key_path = api_key_path.split('/')
            api_key_path = os.path.join(PROJECT_ROOT_DIR, *api_key_path)
        else:
            api_key_path = os.path.join(PROJECT_ROOT_DIR, api_key_path)
        with open(api_key_path, 'r') as f:
            access_token = f.read().rstrip()
        # self.model = AutoModel.from_pretrained('bzantium/NV-Embed-v1', trust_remote_code=True, token=access_token, load_in_4bit=True, device_map="auto")
        super().__init__('bzantium/NV-Embed-v1', trust_remote_code=True, token=access_token)
        self.max_seq_length = 4096
        self.tokenizer.padding_side = "right"

    def add_eos(self, input_examples):
        input_examples = [input_example + self.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def predict(self, pairs):
        # Each query needs to be accompanied by a corresponding instruction describing the task.
        task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question", }
        query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "
        queries, docs = [p[0] for p in pairs], [p[1] for p in pairs]
        query_embeddings = self.encode(self.add_eos(queries), max_length=4096, batch_size=len(queries), prompt=query_prefix, normalize_embeddings=True)
        # query_embeddings = self.model.encode(queries, max_length=4096, batch_size=len(queries), prompt=query_prefix, normalize_embeddings=True)
        doc_embeddings = self.encode(self.add_eos(docs), max_length=4096, batch_size=len(docs), normalize_embeddings=True)
        # doc_embeddings = self.model.encode(docs, max_length=4096, batch_size=len(docs), normalize_embeddings=True)
        return list((query_embeddings @ doc_embeddings.T) * 100)


class BAAILLMEmbedder(SentenceTransformer):
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on {device}")
        model_path = os.path.join(PROJECT_ROOT_DIR, 'models', 'BAAI', 'llm-embedder')
        if not os.path.exists(os.path.join(model_path, "config.json")):
            model_path = "BAAI/llm-embedder"
        super().__init__(model_path, device=device)

    def predict(self, pairs):
        query_embeddings = self.encode([p[0] for p in pairs], normalize_embeddings=True)
        doc_embeddings = self.encode([p[1] for p in pairs], normalize_embeddings=True)
        scores = []
        for q_emb, d_emb in zip(query_embeddings, doc_embeddings):
            similarity = cosine_similarity([q_emb], [d_emb])[0][0] * 100
            scores.append(similarity)
        return scores


class BAAIBiEncoder:
    def __init__(self):
        self.task = 'Given a web search query, retrieve relevant passages that answer the query.'
        examples = [
            {
                'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
                'query': 'how to integrate VoltMX with Salesforce',
                'response': "Integrating VoltMX with Salesforce involves using VoltMX's integration services to connect with Salesforce APIs. This allows seamless data exchange between VoltMX applications and Salesforce, enabling functionalities like retrieving Salesforce data, updating records, and triggering workflows. The integration process typically includes configuring authentication, setting up API endpoints, and mapping data fields to ensure smooth communication between the platforms."
            },
            {
                'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
                'query': 'features of VoltMX Go',
                'response': "VoltMX Go is a low-code platform designed to accelerate app development. Key features include a drag-and-drop interface for building user interfaces, pre-built connectors for integrating with various backend systems, and robust security measures to protect data. VoltMX Go also supports multi-channel deployment, allowing developers to create applications that work seamlessly across web, mobile, and other digital touchpoints."
            },
            {
                'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
                'query': 'Domino query language basics',
                'response': "Domino Query Language (DQL) is used to query data in Domino databases. It allows users to perform complex searches and retrieve specific data sets based on various criteria. Basic syntax includes SELECT statements to specify the fields to retrieve, WHERE clauses to define conditions, and ORDER BY clauses to sort the results. DQL is designed to be intuitive and powerful, enabling efficient data retrieval and manipulation."
            }
        ]

        self.examples = [self.get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
        self.examples_prefix = '\n\n'.join(self.examples) + '\n\n'  # if there not exists any examples, just set examples_prefix = ''

        model_path = os.path.join(PROJECT_ROOT_DIR, 'models', 'BAAI', 'bge-en-icl')
        if not os.path.exists(model_path):
            model_path = "BAAI/bge-en-icl"

        self.query_max_len, self.doc_max_len = 512, 512
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'<instruct>{task_description}\n<query>{query}'

    @staticmethod
    def get_detailed_example(task_description: str, query: str, response: str) -> str:
        return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

    @staticmethod
    def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
        inputs = tokenizer(
            queries,
            max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
                tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
            return_token_type_ids=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=False
        )
        prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
        suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
        new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
        new_queries = tokenizer.batch_decode(inputs['input_ids'])
        for i in range(len(new_queries)):
            new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
        return new_max_length, new_queries

    def predict(self, pairs):
        queries = [self.get_detailed_instruct(self.task, p[0]) for p in pairs]
        documents = [p[1] for p in pairs]
        new_query_max_len, new_queries = self.get_new_queries(queries, self.query_max_len, self.examples_prefix,
                                                              self.tokenizer)

        query_batch_dict = self.tokenizer(new_queries, max_length=new_query_max_len, padding=True, truncation=True,
                                          return_tensors='pt')
        doc_batch_dict = self.tokenizer(documents, max_length=self.doc_max_len, padding=True, truncation=True,
                                        return_tensors='pt')

        with torch.no_grad():
            query_outputs = self.model(**query_batch_dict)
            query_embeddings = self.last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
            doc_outputs = self.model(**doc_batch_dict)
            doc_embeddings = self.last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])

        # normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
        scores = (query_embeddings @ doc_embeddings.T) * 100
        return scores.tolist()

    def encode(self, text):
        doc_batch_dict = self.tokenizer([text], max_length=self.doc_max_len, padding=True, truncation=True,
                                   return_tensors='pt')
        with torch.no_grad():
            doc_outputs = self.model(**doc_batch_dict)
            doc_embeddings = self.last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])
        # normalize embeddings
        return F.normalize(doc_embeddings, p=2, dim=1)


class BAAICrossEncoder(SentenceTransformer):
    def __init__(self):
        super().__init__("BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16})

    def predict(self, pairs):
        instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
        prompt = f'<instruct>{instruction}\n<query>'
        # Compute the query and document embeddings
        query_embeddings = self.encode([p[0] for p in pairs], prompt=prompt)
        doc_embeddings = self.encode([p[1] for p in pairs])
        # Compute the cosine similarity between the query and document embeddings
        similarities = self.similarity(query_embeddings, doc_embeddings)
        return similarities.tolist()


class Stella400MV5(SentenceTransformer):
    def __init__(self):
        super().__init__('dunzhang/stella_en_400M_v5', trust_remote_code=True)

    def predict(self, pairs):
        query_embeddings = self.encode([p[0] for p in pairs], prompt_name="s2p_query")
        doc_embeddings = self.encode([p[1] for p in pairs])
        return self.similarity(query_embeddings, doc_embeddings).tolist()


class Stella1_5BV5(SentenceTransformer):
    def __init__(self):
        super().__init__('dunzhang/stella_en_1.5B_v5', trust_remote_code=True)

    def predict(self, pairs):
        query_embeddings = self.encode([p[0] for p in pairs], prompt_name="s2p_query")
        doc_embeddings = self.encode([p[1] for p in pairs])
        return self.similarity(query_embeddings, doc_embeddings).tolist()


class SFREmbeddingMistral(SentenceTransformer):
    def __init__(self):
        super().__init__('Salesforce/SFR-Embedding-Mistral', load_in_4bit=True)

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def predict(self, pairs):
        # Each query must come with a one-sentence instruction that describes the task
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        queries, docs = [self.get_detailed_instruct(task, p[0]) for p in pairs], [p[1] for p in pairs]

        embeddings = self.encode(queries + docs)
        scores = util.cos_sim(embeddings[:2], embeddings[2:]) * 100
        return scores.tolist()


class SFREmbedding2R(SentenceTransformer):
    def __init__(self):
        super().__init__('Salesforce/SFR-Embedding-2_R', load_in_4bit=True)

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def predict(self, pairs):
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        queries, docs = [self.get_detailed_instruct(task, p[0]) for p in pairs], [p[1] for p in pairs]
        embeddings = self.encode(queries + docs)
        scores = self.similarity(embeddings[:2], embeddings[2:]) * 100
        return scores.tolist()


class TextEmbeddingGecko:
    @staticmethod
    def encode(docs: List[str], prompt:str=None):
        try:
            data = {}
            embed_request = EmbedRequest()
            if docs:
                data = {
                    "instances": [{
                        "content": doc,
                        "task_type": "SEMANTIC_SIMILARITY"
                    } for doc in docs]
                }
                res = embed_request(model_id="textembedding-gecko@003", data=data)
                response = res['predictions']
            else:
                data = {
                    "instances": [{
                        "content": prompt,
                        "task_type": "SEMANTIC_SIMILARITY"
                    }]
                }
                res = embed_request(model_id="textembedding-gecko@003", data=data)
                response = res['predictions'][0]['embeddings']['values']
            assert response, "No Response from Google Embeddings"
            return response
        except Exception as e:
            logger.error(str(e))

    def predict(self, pairs):
        queries, docs = [p[0] for p in pairs], [p[1] for p in pairs]
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(docs)
        return list((query_embeddings @ doc_embeddings.T) * 100)


class MultilingualEmbeddingGecko:
    @staticmethod
    def encode(docs: List[str], prompt:str=None):
        try:
            data = {}
            embed_request = EmbedRequest()
            if docs:
                data = {
                    "instances": [{
                        "content": doc,
                        "task_type": "SEMANTIC_SIMILARITY"
                    } for doc in docs]
                }
                res = embed_request(model_id="textembedding-gecko-multilingual@001", data=data)
                response = res['predictions']
            else:
                data = {
                    "instances": [{
                        "content": prompt,
                        "task_type": "SEMANTIC_SIMILARITY"
                    }]
                }
                res = embed_request(model_id="textembedding-gecko-multilingual@001", data=data)
                response = res['predictions'][0]['embeddings']['values']
            assert response, "No Response from Google Embeddings"
            return response
        except Exception as e:
            logger.error(str(e))

    def predict(self, pairs):
        queries, docs = [p[0] for p in pairs], [p[1] for p in pairs]
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(docs)
        return list((query_embeddings @ doc_embeddings.T) * 100)


class TextMultimodalEmbeddingGecko:
    @staticmethod
    def encode(prompt, base64_encoded_img):
        try:
            data = {
                "instances": [{
                    "text": prompt,
                    "image": {"bytesBase64Encoded": base64_encoded_img}
                }]
            }
            embed_request = EmbedRequest()
            res = embed_request(model_id="multimodalembedding@001", data=data)
            response = res['predictions'][0]['textEmbedding'], res['predictions'][0]['textEmbedding']
            assert response, "No Response from Google Embeddings"
            return response
        except Exception as e:
            logger.error(str(e))
