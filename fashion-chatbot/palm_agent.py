from typing import Any, Dict, List, Optional
from langchain.vectorstores import Pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from palm import PalmLLM


class PalmAgent():
    """
    required fields:
    index: pinecone object
    embed_model: model object
    label: str
    """
    def __init__(self, index, embed_model, label):
        self.index = index
        self.embed_model = embed_model
        self.label = label

        # vectorstore
        vectorstore = Pinecone(
            self.index, self.embed_model.embed_query, self.label
        )

        # chat completion llm
        llm = PalmLLM()
        # conversational memory
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=3,
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        # retrieval qa chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            input_key="input",
            output_key="output"
        )

        # tool
        tools = [
            Tool(
                name='Fashion Knowledge',
                func=self.run_qa_chain,
                description=(
                    # 'use this tool when giving clothing or fashion recommendations'
                    'use this tool when checking the availability of fashion items or when recommending available fashion items'
                )
            )
        ]

        # agent
        self.agent = initialize_agent(
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=5,
            early_stopping_method='generate',
            memory=conversational_memory,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )


    def run_qa_chain(self, query):
        output = self.qa_chain({"input": query}, return_only_outputs=True)
        return output

    def __call__(self, query):
        final_step = []
        contained = []
        res = ""
        try:
            response = self.agent({"input": query})
            print(response)
            final_step = response['intermediate_steps'][-1] if response['intermediate_steps'] else []
            source_docs = final_step[1]['source_documents'] if final_step else []
            contained =  [{self.label: x.page_content, 'metadata': x.metadata} for x in [d for d in source_docs if source_docs]]
            res = response["output"]

        except Exception as e:
            res = f"Error: {str(e)}"
            if final_step:
                source_docs = final_step[1]['source_documents'] if final_step else []
                contained = [{self.label: x.page_content, 'metadata': x.metadata} for x in
                             [d for d in source_docs if source_docs]]
                res = final_step[1]["output"]

        return res, contained
