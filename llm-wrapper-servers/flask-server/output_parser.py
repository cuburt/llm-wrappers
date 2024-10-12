
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


class Sentiment(BaseModel):
  sentiment: str = Field(description="the sentiment, as a unique string and enclosed with double quotes.")
  score: str = Field(description="the probability score, as a unique string and enclosed with double quotes.")

class Sentiments(BaseModel):
  sentiments: List[Sentiment] = Field(description="List of sentiments")

class SentimentParser:

    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=Sentiments)

        format_instructions = self.parser.get_format_instructions()
        print("SENTIMENT: ", format_instructions)
        template_string = """You are an expert when it comes to calculating probabilities of sentiments from texts. \
                You will be given a text and you will return a list of these sentiments: {sentiments} with calculation of probability score for each. \
                The overall probability scores must have a total of 1.00

                Text: {query}

                {format_instructions}
                """


        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["query", "sentiments"],
            partial_variables={"format_instructions": format_instructions}
        )


class Keyword(BaseModel):
    keyword: str = Field(description="the keyword, as a unique string and enclosed with double quotes.")
    score: str = Field(
        description="the maximum marginal likelihood score, as a unique string and enclosed with double quotes.")


class Keywords(BaseModel):
    keywords: List[Keyword] = Field(description="List of keywords")


class KeywordParser:

    def __init__(self):

        self.parser = PydanticOutputParser(pydantic_object=Keywords)

        format_instructions = self.parser.get_format_instructions()
        print("KEYWORD: ", format_instructions)
        template_string = """You are an expert when it comes to extracting keywords from texts and calculating their maximum marginal likelihood. \
                You will be given a text and you will extract {num_keys} keywords. Calculate the maximum marginal likelihood for each keyword. \
                The overall maximum marginal likelihood must have a total of 1.00

                Text: {query}

                {format_instructions}
                """

        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["query", "num_keys"],
            partial_variables={"format_instructions": format_instructions}
        )


class Tone(BaseModel):
  tone: str = Field(description="the tone, as a unique string and enclosed with double quotes.")
  score: str = Field(description="the probability score, as a unique string and enclosed with double quotes.")


class Tones(BaseModel):
  tones: List[Tone] = Field(description="List of tones")

class ToneParser:

    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=Tones)

        format_instructions = self.parser.get_format_instructions()
        print("TONE: ", format_instructions)
        template_string = """You are an expert when it comes to calculating probability distribution of a list of tones from texts. \
                The best tone from the list gets the highest probability, while the worst gets the lowest. The overall probability scores must ALWAYS add up to 1.00 \
                You will be given a text and you will return a list of these tones: {tones}, with calculation of probability score for each. \


                Text: {query}

                {format_instructions}
                """

        self.prompt = PromptTemplate(
            template=template_string,
            input_variables=["query", "tones"],
            partial_variables={"format_instructions": format_instructions}
        )

