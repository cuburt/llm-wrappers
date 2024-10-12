import json
from flask import Flask, request
from llm import PalmLLM, GeminiLLM
from output_parser import SentimentParser, KeywordParser, ToneParser
import re

app = Flask(__name__)


def format_query(query, sentiments=None, num_keys=None, tones=None):
    """
    Formats query with a template instruction.
    """
    formatted_output = ""
    if num_keys:
        parser = KeywordParser()
        formatted_output = parser.prompt.format(query=query, num_keys=str(num_keys))
    if sentiments:
        parser = SentimentParser()
        formatted_output = parser.prompt.format(query=query, sentiments=sentiments)
    if tones:
        parser = ToneParser()
        formatted_output = parser.prompt.format(query=query, tones=tones)

    return formatted_output


def format_request_payload(request, key, values=None):
    return request.get(key, values)


@app.route('/models/<model>/tasks/<task>:predict', methods=['POST'])
def predict(model, task):
    try:
        llm = PalmLLM()

        if model == 'gemini':
            llm = GeminiLLM()

        text = format_request_payload(request.get_json(force=True), 'text')
        _output = ""

        if task == 'keyphrase-extraction':
            num_keys = format_request_payload(request.get_json(force=True),
                                              'num_keys',
                                              5)
            response_palm = llm(format_query(query=text, num_keys=num_keys))
            parser = KeywordParser()
            try:
                parsed_output = parser.parser.parse(response_palm)
            except:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", response_palm)
                parsed_output = parser.parser.parse(parsed_response_palm)
            _output = [{"keyword": k.keyword, "score": k.score} for k in parsed_output.keywords]

        if task == 'sentiment-analysis':
            sentiments = format_request_payload(request.get_json(force=True),
                                                'sentiments',
                                                ["Highly Positive", "Positive", "Neutral", "Negative", "Highly Negative"])

            response_palm = llm(format_query(query=text, sentiments=', '.join(sentiments)))
            parser = SentimentParser()
            try:
                parsed_output = parser.parser.parse(response_palm)
            except:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", response_palm)
                parsed_output = parser.parser.parse(parsed_response_palm)
            _output = [{"sentiment": s.sentiment, "score": s.score} for s in parsed_output.sentiments]

        if task == 'emotion-analysis':
            tones = format_request_payload(request.get_json(force=True),
                                           'tones',
                                           ['Sadness', 'Joy', 'Surprise', 'Disgust', 'Fear', 'Anger'])
            response_palm = llm(format_query(query=text, tones=', '.join(tones)))
            parser = ToneParser()
            try:
                parsed_output = parser.parser.parse(response_palm)
            except:
                parsed_response_palm = re.sub(r"(?<!\\)\\'", "'", response_palm)
                parsed_output = parser.parser.parse(parsed_response_palm)
            _output = [{"tone": t.tone, "score": t.score} for t in parsed_output.tones]

        if task == 'summariser':
            _output = llm("Provide a summary with about two sentences for the following article: " + text)

        response = json.dumps({"predictions": _output})

    except Exception as e:
        response = json.dumps({"Error": str(e)})

    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)