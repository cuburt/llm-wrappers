# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:39:30 2024

@author: manjunath_hegde, cuburt.balanon

@project: XAI

"""
import os
from typing import List, Optional, Any
from semantic_router import Route
from scripts.log import logger
from scripts.server.encoders import (TextEmbeddingGecko,
                                     BAAICrossEncoder,
                                     BAAIBiEncoder,
                                     BAAILLMEmbedder,
                                     Stella400MV5,
                                     Stella1_5BV5)
from semantic_router.encoders import CohereEncoder, BaseEncoder, HuggingFaceEncoder, OpenAIEncoder
from semantic_router.layer import RouteLayer
import platform
from pathlib import Path
import pandas as pd
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent)  # set project root directory


def read_api_key(service_provider):
    """
    Options: "openai", "nvidia", "cohere"
    """
    api_key_path = f"scripts/server/.{service_provider}_api_key.txt"
    if platform.system() == 'Windows':
        api_key_path = api_key_path.split('/')
        api_key_path = os.path.join(PROJECT_ROOT_DIR, *api_key_path)
    else:
        api_key_path = os.path.join(PROJECT_ROOT_DIR, api_key_path)
    with open(api_key_path, 'r') as f:
        access_token = f.read().rstrip()
    if not access_token:
        return f"Error 401: No {service_provider} API-KEY saved."

    return access_token


class TextEmbeddingGeckoRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "google"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = TextEmbeddingGecko()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = self.client.encode(docs=docs)
        return [prediction["embeddings"]["values"] for prediction in predictions]


class Stella400MV5Router(BaseEncoder):
    client: Optional[Any] = None
    type: str = "stella400m"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = Stella400MV5()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class Stella1_5BV5Router(BaseEncoder):
    client: Optional[Any] = None
    type: str = "stella5b"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = Stella1_5BV5()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class MsMarcoMiniLML6V2Router(HuggingFaceEncoder):
    name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    score_threshold: float = 0.95


class BAAICrossEncoderRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "baaicrossencoder"

    def __init__(
            self,
            name: Optional[str] = type,
            score_threshold: float = 0.75,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = BAAICrossEncoder()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class BAAIBiEncoderRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "baaibiencoder"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = BAAIBiEncoder()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class BAAILLMEmbedderRouter(BaseEncoder):
    client: Optional[Any] = None
    type: str = "baaillmembedder"

    def __init__(
        self,
        name: Optional[str] = type,
        score_threshold: float = 0.75,
    ):
        super().__init__(name=name, score_threshold=score_threshold)
        self.client = BAAILLMEmbedder()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        predictions = [self.client.encode(doc) for doc in docs]
        return predictions


class SemanticRouter:
    def __init__(self):
        try:
            feedback_csv = "scripts/server/feedback.csv"
            feedback_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, *feedback_csv.split('/')))
            voltmx_utterances_from_feedback = [str(v) for v in feedback_df["query"].values]
            voltmx = Route(
                name="voltmx",
                utterances=[
                    "how to create a component in hcl volt iris?",
                    "how create a component in volt mx?",
                    "how to make a https call in voltmx?",
                    "get started with iris",
                    "invoke foundry in within iris",
                    "how to add a button to form in voltmx?",
                    "how to add a button to form?",
                    "How do I create a new project in VoltMX Iris?",
                    "Show me the steps to import an existing project.",
                    "Can you guide me on how to set up VoltMX Iris for the first time?",
                    "How can I customize widgets in VoltMX Iris?",
                    "How can I debug my code in VoltMX Iris?",
                    "Explain how to use the collaboration tools in VoltMX Iris.",
                    "What are common issues with VoltMX Iris and how can I fix them?",
                    "how to create a component?",
                    "is voltmx iris supported on linux?",
                    "how to make a https call",
                    "code snippet to make a http call",
                    "code snippet to hit imdb to get the list of top 10 movies of given year and display then in list",
                    "how to persist the data in voltmx?",
                    "how to add animation to button",
                    "How to add action",
                    "how to add google login?",
                    "how to add a button to form?",
                    "What are the key components of VoltMX?",
                    "What programming languages are used in VoltMX development?",
                    "How do I design a user interface?",
                    "What is VoltMX Foundry, and how does it work?",
                    "How do I integrate backend services with my VoltMX app?",
                    "What are the different types of widgets available in VoltMX?",
                    "How do I handle navigation between forms",
                    "What is the best way to debug a VoltMX application?",
                    "How do I manage application themes and skins",
                    "How can I deploy my VoltMX app?",
                    "What are VoltMX Widgets and how do they differ from standard HTML elements?",
                    "How do I implement data binding?",
                    "Can I use third-party libraries in my VoltMX project?",
                    "How do I handle user authentication?"
                    "What is the process for updating an existing VoltMX app?",
                    "How do I use VoltMX App Services for push notifications?",
                    "What support options are available for VoltMX developers?",
                    "How do I manage app configurations for different environments?",
                    "How do I manage user roles and permissions",
                    "volt mx", "mx", "voltscript"
                ] + voltmx_utterances_from_feedback)

            go = Route(
                name="go",
                utterances=[
                    "how to use VoltFormula in Volt MX Go?",
                    "how to configure mobile app browser using go?",
                    "explain rosetta frow overview in voltmx go?",
                    "Explain how to migrate a mainframe application to a modern platform.",
                    "go", "volt mx go", "mx go"], )

            domino = Route(
                name="domino",
                utterances=[
                    "how to create a form in domino?",
                    "what are custom controls in domino?",
                    "How to create a database in Domino?",
                    "How to delete a page in Domino?",
                    "what are framesets in Domino?",
                    "Show me how to use LotusScript in my applications",
                    "domino", "LotusScript"], )

            routes = [voltmx, go, domino]

            try:
                logger.info("Initializing router...")
                encoder = OpenAIEncoder(name="text-embedding-3-large", openai_api_key=read_api_key("openai"))
                # encoder = BAAILLMEmbedderRouter()
                self.routelayer = RouteLayer(encoder=encoder, routes=routes)
                assert encoder, "API Key expired or invalid."
                assert self.routelayer, "API Key expired or invalid."

            except (Exception, AssertionError) as e:
                logger.info("Router initialization failed. Reverting to GoogleRouter...")
                logger.warning(str(e))
                encoder = TextEmbeddingGeckoRouter()
                self.routelayer = RouteLayer(encoder=encoder, routes=routes)
            logger.info("Router successfully initialised.")
        except Exception as e:
            raise Exception(str(e))

    def __call__(self, prompt):
        return self.routelayer(prompt).name
