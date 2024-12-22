# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon,manjunath_hegde

@project: XAI

@input:

@output:

@des
"""
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class IQPromptTemplate:
    def __init__(self, response_format: str = ""):
        template_string = """
You are an expert in VoltMX, VoltMX Go, and Domino.
Please follow these instructions:

1. Provide a comprehensive, detailed, and human-like answer to the question below, while being clear and procedural.
2. Use the 'page_content' in the context to understand the topic and support your answer. \
If there is no context, ask the user to clarify the question.
3. When asked for code snippet, please provide the code snippet along with other relevant answer.
4. Do not reveal that you've been provided with context or documents.
5. Always cite and recommend additional sources accurately, using only the 'source' in the context.
6. Make reference to the Chat History when citing previous queries and responses to maintain context continuity.
7. Ensure that the output adheres to any specified format instructions provided.
8. AVOID "kony" references. ALWAYS use "voltmx" inplace of "kony" when generating code that uses voltmx API.


Chat History:
{chat_history}

Question:
{input}

Context:
{context}

{format_instructions}

Helpful AI Response:
"""

        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["context", "input", "chat_history"],
            partial_variables={"format_instructions": f"format Instructions:\n{response_format}"}
        )


class MemoryPromptTemplate:
    
    def __init__(self):
        template = """\
Given a chat history and the latest user question which might reference context in the chat history, \
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

        self.prompt = ChatPromptTemplate.from_messages([("system", template),
                                                                           MessagesPlaceholder("chat_history"),
                                                                           ("human", "{input}")])

class DocumentParserPromptTemplate:
    def __init__(self):
        template = "page_content: {page_content}\nsource: {source}"

        self.prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template=template
        )
