from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from llm import GPT4oMiniLLM, Gemini15LLM
from retriever import Retriever
from vectorstore import Vectorstore
import base64
import io
from PIL import Image
from operator import itemgetter


class Chain:
    def __init__(self):
        self.retriever = Retriever(vectorstore=Vectorstore().vectorstore, k=3)
        self.gpt_chain = (
                {
                    "context": self.retriever | RunnableLambda(self.split_image_text_types),
                    "input": RunnablePassthrough(),
                }
                | RunnableParallel(
            {"response": RunnableLambda(self.openai_prompt_func) | GPT4oMiniLLM(), "context": itemgetter("context")})
        )
        self.gemini_chain = (
                {
                    "context": self.retriever | RunnableLambda(self.split_image_text_types),
                    "input": RunnablePassthrough(),
                }
                | RunnableParallel({"response": RunnableLambda(self.gemini_prompt_func) | Gemini15LLM(), "context": itemgetter("context")})
        )

    @staticmethod
    def resize_base64_image(base64_string, size=(128, 128), filetype=None):
        """
        Resize an image encoded as a Base64 string.

        Args:
        base64_string (str): Base64 string of the original image.
        size (tuple): Desired size of the image as (width, height).

        Returns:
        str: Base64 string of the resized image.
        """
        # Decode the Base64 string
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        # Resize the image
        resized_img = img.resize(size, Image.LANCZOS)
       # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)

        # Encode the resized image to Base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def is_base64(s):
        """Check if a string is Base64 encoded"""
        try:
            return base64.b64encode(base64.b64decode(s)) == s.encode()
        except Exception:
            return False

    def split_image_text_types(self, docs):
        """Split numpy array images and texts"""
        images = []
        text = []
        for doc in docs:
            if doc.metadata and "base64_image" in doc.metadata and self.is_base64(doc.metadata["base64_image"]):
                # Resize image to avoid OAI server error
                images.append(doc.page_content)  # base64 encoded str
            else:
                text.append(doc.page_content)
        return {"images": images, "texts": text}

    def openai_prompt_func(self, data_dict):
        messages = []
        question = ""
        # Adding image(s) to the messages if present
        if "context" in data_dict and data_dict["context"]["images"] and data_dict['input']['question']:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image.split(',')[0]},{self.resize_base64_image(image.replace('data:image/jpeg;base64,', '').replace('data:image/png;base64,', ''))}"
                    },
                }
                messages.append(image_message)

            question = data_dict['input']['question']
            # Adding the text message for analysis
            text_message = {
                "type": "text",
                "text": (
                    "As an expert User Interface critic, your task is to generate a new design based on the user-provided query and UI images provided, then generate HTML/Javascript code for it."
                    f" User-provided query: {question}\n\n"
                ),
            }
            messages.append(text_message)

        if "image" in data_dict['input'] and data_dict['input']['image']:
            image_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"{data_dict['input']['image'].split(';')[0]};base64,{self.resize_base64_image(data_dict['input']['image'].replace('data:image/jpeg;base64,', '').replace('data:image/png;base64,', ''))}"
                },
            }
            messages.append(image_message)
            text_message = {
                "type": "text",
                "text": (
                    "As an expert User Interface critic, your task is to analyze and interpret the UI image, suggest improvements, and generate HTML/Javascript code for it."
                ),
            }
            messages.append(text_message)

        return [HumanMessage(content=messages)]

    def gemini_prompt_func(self, data_dict):
        parts = []
        question = ""
        # Adding image(s) to the messages if present
        if "context" in data_dict and data_dict["context"]["images"] and data_dict['input']['question']:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "inlineData": {
                        "mimeType": image.split(';')[0].replace('data:', ''),
                        "data": self.resize_base64_image(image.replace('data:image/jpeg;base64,', '').replace('data:image/png;base64,', ''))
                    },
                }
                parts.append(image_message)

            question = data_dict['input']['question']
            # Adding the text message for analysis
            text_message = {
                "text": (
                    "As an expert User Interface critic, your task is to generate a new design based on the user-provided query and UI images provided, then generate HTML/Javascript code for it."
                    f" User-provided query: {question}\n\n"
                )
            }
            parts.append(text_message)

        elif "image" in data_dict['input'] and data_dict['input']['image']:
            image_message = {
                "inlineData": {
                    "mimeType": data_dict['input']['image'].split(';')[0].replace('data:', ''),
                    "data": self.resize_base64_image(data_dict['input']['image'].replace('data:image/jpeg;base64,', '').replace('data:image/png;base64,', ''))
                },
            }
            parts.append(image_message)
            text_message = {
                "text": (
                    "As an expert User Interface critic, your task is to analyze and interpret the UI image, suggest improvements, and generate HTML/Javascript code for it."
                ),
            }
            parts.append(text_message)

        return [HumanMessage(content=parts)]

    def __call__(self, question=None, image=None, model="gemini"):
        chain = eval(f"self.{model}_chain")
        output = chain.invoke({"question": question, "image": image})
        response, similar_images = output["response"], output["context"]["images"]
        print(similar_images)
        # similar_images = self.retriever.invoke(question if question else image)
        return response, [self.retriever.vectorstore._decode_base64_to_image(i.split(",")[1]) for i in similar_images]