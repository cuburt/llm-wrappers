import os
from pathlib import Path
from pipeline import CodeInterpreterPipeline, CodeAssistantPipeline
import nest_asyncio
from fastapi import FastAPI
import uvicorn
import document_loader

nest_asyncio.apply()
server = FastAPI()

VOLTMX_PATH = os.path.join(str(Path(__file__).parent.parent), "data", "compiled_data.csv")
VOLTSCRIPT_PATH = os.path.join(str(Path(__file__).parent.parent), "data", "voltscript.csv")
LOTUSCRIPT_PATH = os.path.join(str(Path(__file__).parent.parent), "data", "lotusscript_classes.csv")
LOTUSCRIPT2_PATH = os.path.join(str(Path(__file__).parent.parent), "data", "lotusscript_classes_2.csv")
LOTUSCRIPT3_PATH = os.path.join(str(Path(__file__).parent.parent), "data", "lotusscript_tutorials.csv")
VOLTMX_DATA = document_loader.column_in_csv_to_list_of_docs(file=VOLTMX_PATH, col="Content",
                                                                metadata_cols=['Filename'])
VOLTSCRIPT_DATA = document_loader.column_in_csv_to_list_of_docs(file=VOLTSCRIPT_PATH, col="code",
                                                                metadata_cols=['filename'])
LOTUSCRIPT_DATA = document_loader.column_in_csv_to_list_of_docs(file=LOTUSCRIPT_PATH, col="Text")
LOTUSCRIPT2_DATA = document_loader.crawled_csv_to_list_of_docs(file=LOTUSCRIPT2_PATH)
LOTUSCRIPT3_DATA = document_loader.column_in_csv_to_list_of_docs(file=LOTUSCRIPT3_PATH, col="Text")

DATA = VOLTMX_DATA + VOLTSCRIPT_DATA + LOTUSCRIPT_DATA + LOTUSCRIPT2_DATA + LOTUSCRIPT3_DATA
CHAT_PIPELINE = CodeAssistantPipeline(DATA)
CODE_PIPELINE = CodeInterpreterPipeline(DATA)


@server.post("/domino/copilot:upload")
async def domino_upload(request:dict):
    try:
        """
        To insert new documents: 
        vectorstore.insert_docs(List[Dict])
        or:
        vectorstore.add_documents(List[Documents])

        Examples:
        voltmx_vectorstore.insert_docs([{"text": text, "metadata": metadata}])
        voltmx_vectorstore.add_documents(docloader.txt_to_list_of_docs(dir="your_directory_to_textfiles"))
        """
        response = CODE_PIPELINE.vectorstore.insert_docs(request.get('data'))
        return {"answer": ''.join([r for r in response])}
    except Exception as e:
        return {"Error": str(e)}

@server.post("/domino/copilot/models/{model}:translate")
async def domino_translate(model: str, request: dict):
    try:
        response, sandbox_output, source_docs = CODE_PIPELINE(
            query=f"Translate the following code to {request.get('target_lang')}: {request.get('query')}")
        return {"answer": response, "sandbox-output": sandbox_output, "reference": source_docs if ('return_references' in request and request.get('return_references')) else []}

    except Exception as e:
        return {"Error": str(e)}


@server.post("/domino/copilot/models/{model}:instruct")
async def domino_instruct(model: str, request: dict):
    try:
        response, sandbox_output, source_docs = CODE_PIPELINE(query=request.get("query"))
        return {"answer": response, "sandbox-output": sandbox_output, "reference": source_docs if ('return_references' in request and request.get('return_references')) else []}

    except Exception as e:
        return {"Error": str(e)}


@server.post("/domino/copilot/models/{model}:annotate")
async def domino_annotate(model: str, request: dict):
    try:
        response, sandbox_output, source_docs = CODE_PIPELINE(query=f"Generate comments in this code: {request.get('query')}")
        return {"answer": response, "sandbox-output": sandbox_output, "reference": source_docs if ('return_references' in request and request.get('return_references')) else []}

    except Exception as e:
        return {"Error": str(e)}

@server.post("/voltiq/copilot:upload")
async def voltiq_upload(request: dict):
    try:
        """
        To insert new documents: 
        vectorstore.insert_docs(List[Dict])
        or:
        vectorstore.add_documents(List[Documents])

        Examples:
        voltmx_vectorstore.insert_docs([{"text": text, "metadata": metadata}])
        voltmx_vectorstore.add_documents(docloader.txt_to_list_of_docs(dir="your_directory_to_textfiles"))
        """
        response = CHAT_PIPELINE.vectorstore.insert_docs(request.get('data'))
        return {"answer": ''.join([r for r in response])}
    except Exception as e:
        return {"Error": str(e)}

@server.post("/voltiq/copilot/models/{model}:translate")
async def voltiq_translate(model: str, request: dict):
    try:
        response, sandbox_output, source_docs = CHAT_PIPELINE(
            query=f"Translate the following code to {request.get('target_lang')}: {request.get('query')}")
        return {"answer": response, "sandbox-output": sandbox_output, "reference": source_docs if ('return_references' in request and request.get('return_references')) else []}
    except Exception as e:
        return {"Error": str(e)}


@server.post("/voltiq/copilot/models/{model}:instruct")
async def voltiq_instruct(model: str, request: dict):
    try:
        response, sandbox_output, source_docs = CHAT_PIPELINE(query=request.get("query"))
        return {"answer": response, "sandbox-output": sandbox_output, "reference": source_docs if ('return_references' in request and request.get('return_references')) else []}
    except Exception as e:
        return {"Error": str(e)}


@server.post("/voltiq/copilot/models/{model}:annotate")
async def voltiq_annotate(model: str, request: dict):
    try:
        response, sandbox_output, source_docs = CHAT_PIPELINE(query=f"Generate comments in this code: {request.get('query')}")
        return {"answer": response, "sandbox-output": sandbox_output, "reference": source_docs if ('return_references' in request and request.get('return_references')) else []}
    except Exception as e:
        return {"Error": str(e)}



if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8080)
