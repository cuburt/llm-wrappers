# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from scripts.log import logger


class CodeInterpreterSchema:

    def __init__(self):
        self.bot_response = ResponseSchema(
            name="bot_response",
            description="the human-like response, as a unique string and ALWAYS enclosed with double quotes.",
        )

        self.code = ResponseSchema(
            name="code",
            description="the programming code generated when asked to translate or generate code, as a unique string and ALWAYS enclosed with double quotes. New lines or '\n' are unescaped.",
        )

        self.output_parser = StructuredOutputParser.from_response_schemas(
            [self.bot_response, self.code]
        )

        # self.response_format = self.output_parser.get_format_instructions()
        try:

            response_format = """\
    The output should be a markdown code snippet strictly formatted in the following schema, including the leading and \
    trailing "```json" and "```". Keys and values are ALWAYS enclosed with double quotes:
    
    ```json
    {
        "bot_response": string  // the human-like response, as a unique string and ALWAYS enclosed with double quotes.
        "code": string  // the programming code generated when asked to translate or generate code, as a unique string and ALWAYS enclosed with double quotes.
    }
    ```
    
    For python code, enclose code with leading and trailing "```python" and "```".
    For javascript code, enclose code with leading and trailing "```js" and "```".
    
    Regardless of output, DO NOT escape new lines or "\\n".
            """
    
            template_string = """\
    You are an expert when it comes to translating programming languages to another. You will be given an instruction and \
    you will start with a human-like greeting or response and ALWAYS SEPARATE THE CODE if there is a code.
    
    Instruction: {query}
    
    {format_instructions}
            """
    
            
            self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["query"],
            partial_variables={"format_instructions": response_format}        )
        except Exception as e:
            logger.error(str(e))

        # self.prompt = ChatPromptTemplate.from_template(
        #     "{query}.\n Answer as human-like as possible and separate the code if there is a code.\n {format_instructions}")
