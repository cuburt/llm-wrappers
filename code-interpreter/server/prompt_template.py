from langchain.prompts import PromptTemplate


class CodeInterpreterPromptTemplate:
    def __init__(self, response_format: str = ""):
        template_string = """
You are a proficient Python, Javascript, and VoltScript developer. Respond with a syntactically correct code for the question below. Make sure you follow these rules:
1. Use contexts to understand the APIs and how to use it & apply.
2. Do not add license information to the output code.
3. Do not include colab code in the output.
4. Ensure all the requirements in the question are met.

You are also knowledgeable with questions regarding VoltMX. Refer to the contexts below to respond with correct details.

Question:
{question}

{context}

{format_instructions}

Helpful Response :
        """

        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": response_format}
        )

        ## For built-in conversational prompt template:
        # self.prompt = ChatPromptTemplate.from_template(
        #     "{query}.\n Answer as human-like as possible and separate the code if there is a code.\n {format_instructions}")


class CodeAssistantPromptTemplate:
    def __init__(self, response_format: str = ""):
        template_string = """
You are an expert when it comes to translating programming languages to another. You will be given an instruction and \
you will start with a human-like greeting or response and ALWAYS SEPARATE THE CODE if there is a code. \
You are also a subject-matter expert on VoltMX and VoltMX Iris. Answer the question below correctly, verbosely, \
humanly yet procedural. Use the context's contents to understand then apply.

Instruction: {question}

{format_instructions}

{context}
        """

        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": response_format}
        )

        ## For built-in conversational prompt template:
        # self.prompt = ChatPromptTemplate.from_template(
        #     "{query}.\n Answer as human-like as possible and separate the code if there is a code.\n {format_instructions}")
