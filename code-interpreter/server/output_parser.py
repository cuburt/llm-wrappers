from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class CodeInterpreterSchema:

    def __init__(self):
        self.bot_response = ResponseSchema(
            name="non-code",
            description="the human-like response, as a unique string and ALWAYS enclosed with double quotes.",
        )

        self.code = ResponseSchema(
            name="code",
            description="the programming code generated when asked to translate or generate code, as a unique string and ALWAYS enclosed with double quotes. New lines or '\n' are unescaped.",
        )

        self.output_parser = StructuredOutputParser.from_response_schemas(
            [self.bot_response, self.code]
        )
        ## Default response format from output_parser
        self.response_format = self.output_parser.get_format_instructions()
