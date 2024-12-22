import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
# from torch import cuda, backends
import pinecone
import logging
from tqdm.auto import tqdm
from clip_embeddings import ClipEmbeddings
# from llama_agent import LlamaAgent
from palm_agent import PalmAgent
from gemini_agent import GeminiAgent

class Backend:

    def __init__(self):

        dataset_path = "archive/"
        self.df = pd.read_csv(dataset_path+"reference_dataframe.csv", index_col=0)
        self.label = 'productDisplayName'
        self.images = [Image.open(r"archive/images/" + self.df["image"][i]) for i in self.df[:self.df.shape[0]].index]
        self.load()

    def load(self):
        # device = f'cuda:{cuda.current_device()}' if cuda.is_available() else (
        #     "mps" if backends.mps.is_available() else "cpu")
        self.device = "cpu"
        self.embed_model = ClipEmbeddings(model_name="openai/clip-vit-base-patch32", device=self.device)

        logging.info("initialising vector database...")
        pinecone.init(api_key='134e9182-fa0f-4e93-9f16-a01780242c65', environment='gcp-starter')
        self.index = pinecone.Index('multimodal-fashion-small')
        # logging.info("initialising LLAMA agent...")
        # self.llama_agent = LlamaAgent(index=self.index,
        #                               label=self.label,
        #                               embed_model=self.embed_model,
        #                               model_id='meta-llama/Llama-2-13b-chat-hf',
        #                               device=self.device,
        #                               hf_auth='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')

        self.palm_agent = PalmAgent(index=self.index,
                                    label=self.label,
                                    embed_model=self.embed_model)

        self.gemini_agent = GeminiAgent(index=self.index,
                                    label=self.label,
                                    embed_model=self.embed_model)

    def prepare_dataframe(self, dataset_path):
        logging.info("preparing reference data...")
        df = pd.read_csv(dataset_path + "styles.csv", nrows=15000, on_bad_lines="skip")
        df['image'] = df["id"].apply(lambda image: self.check_image_exists(str(image) + ".jpg", dataset_path))
        df = df.dropna(subset=['image'])
        df = df.reset_index(drop=True)
        df.reset_index(inplace=True)
        df.drop(["index"], axis=1, inplace=True)
        return df

    @staticmethod
    def get_all_filenames(directory):
        """
        Returns a set of all filenames in the given directory.
        """
        filenames = {entry.name for entry in os.scandir(directory) if entry.is_file()}
        return filenames

    def check_image_exists(self, image_filename, dataset_path):
        """
        Checks if the desired filename exists within the filenames found in the given directory.
        Returns True if the filename exists, False otherwise.
        """
        # global images
        images = self.get_all_filenames(dataset_path + "images/")
        if image_filename in images:
            # print(image_filename)
            return image_filename
        else:
            return np.nan

    def train(self, batch_size: int = 1):

        for i in tqdm(range(0, len(self.images), batch_size)):
            # select batch of images
            image = self.images[i:i + batch_size]
            # process and resize
            image = self.processor(
                text=None,
                images=image,
                return_tensors='pt',
                padding=True
            )['pixel_values'].to(self.device)
            # get image embeddings
            embedding = self.model.get_image_features(pixel_values=image)
            # batch_emb = np.squeeze(0)
            embedding = embedding.cpu().detach().numpy()
            metadata = {str(key): str(self.df[key][i]) for key in self.df.columns if
                        key in ["productDisplayName", "gender", "masterCategory", "subCategory", "articleType",
                                "baseColour", "season", "year", "usage"]}
            self.index.upsert([(f"{i}", embedding[0].tolist(), metadata)])

@st.cache_resource
def load_backend():
    backend_obj = Backend()
    return backend_obj

if __name__ == "__main__":

    backend_obj = load_backend()

    logging.info("starting app...")
    st.title("Fashion Agent")
    sidetab0, sidetab1, sidetab2 = st.sidebar.tabs(["LLM", "Few-Shot Examples", "About"])

    with sidetab0:
        agent = st.radio("Choose an agent", ["Gemini-Pro", "PaLM-2", "LLaMA-2"],
                         captions=["Agent powered by Gemini-Pro model API",
                                   "Agent powered by PaLM-2's chat-bison model API",
                                   "Agent powered by LLaMA-2's distilled 13b-chat-hf model"])

    with sidetab1:
        st.caption("Example 1")
        user1 = st.text_input(value="Im searching for a stylish jacket.", label="user", key="user1")
        bot1 = st.text_input(value="Certainly, here are some fashionable jackets available in our collection.", label="bot", key="bot1")
        st.divider()

        st.caption("Example 2")
        user2 = st.text_input(value="I want to buy a new summer dress.", label="user", key="user2")
        bot2 = st.text_input(value="Of course, here are some lovely summer dresses we have in stock.",
                             label="bot", key="bot2")
        st.divider()

        st.caption("Example 3")
        user3 = st.text_input(value="Can you help me find a comfortable pair of sneakers?", label="user", key="user3")
        bot3 = st.text_input(value="Absolutely, here are some comfortable sneakers from our selection.",
                             label="bot", key="bot3")
        st.divider()

        st.caption("Example 4")
        user4 = st.text_input(value="Im in need of a formal shirt for an upcoming event.", label="user", key="user4")
        bot4 = st.text_input(value="Certainly, here are some formal shirts that might be suitable for your event.",
                             label="bot", key="bot4")
        st.divider()

        st.caption("Example 5")
        user5 = st.text_input(value="What goes well with a white button-down shirt?", label="user", key="user5")
        bot5 = st.text_input(value="It can go well with navy chinos, khakis, or jeans. You can also pair it with a blazer or cardigan for a more formal look.",
                             label="bot", key="bot5")
        st.divider()

        st.caption("Example 6")
        user6 = st.text_input(value="How should I style a denim jacket?", label="user", key="user6")
        bot6 = st.text_input(value="You can wear it over a t-shirt, blouse, or sweater. You can also layer it with a flannel shirt or hoodie. For pants, you can wear jeans, chinos, or shorts.",
                             label="bot", key="bot6")
        st.divider()

        st.caption("Example 7")
        user7 = st.text_input(value="What are some good outfits for a date?", label="user", key="user7")
        bot7 = st.text_input(value="It depends on the type of date and your personal style. But some classic options include a button-down shirt and chinos, a dress shirt and slacks, or a skirt or dress with a blouse.",
                             label="bot", key="bot7")
        st.divider()

        st.caption("Example 8")
        user8 = st.text_input(value="How should I style a turtleneck sweater?", label="user", key="user8")
        bot8 = st.text_input(value="You can wear it with a button-down shirt or blouse, or on its own with jeans or chinos. You can also layer it with a blazer or cardigan.",
                             label="bot", key="bot8")
        st.divider()

        st.caption("Example 9")
        user9 = st.text_input(value="I need a black puffer jacket for women.", label="user", key="user9")
        bot9 = st.text_input(
            value="We have Zara Womens Black Puffer Jacket in stock. It is a classic and versatile jacket that can be dressed up or down. You can pair it with a dress, jeans, or leggings.",
            label="bot", key="bot9")

        st.divider()

        st.caption("Example 10")
        user10 = st.text_input(value="I need a pair of blue jeans and white sneakers.", label="user", key="user10")
        bot10 = st.text_input(
            value="We have H&M Womens Blue Jeans and Converse Chuck Taylor All Star Low Top Sneakers in stock. These are two classic pieces that can be dressed up or down. You can pair them with a t-shirt, blouse, or sweater.",
            label="bot", key="bot10")

        st.divider()

        st.caption("Example 11")
        user11 = st.text_input(value="What goes well with sandals and a bag for a night out?", label="user", key="user11")
        bot11 = st.text_input(
            value="We have Steve Madden Womens Lindsay Sandals and Michael Kors Womens Bradshaw Bag in stock. These are two stylish and versatile pieces that will help you look your best for a night out. You can pair the sandals with a dress or skirt, and the bag will hold all your essentials.",
            label="bot", key="bot11")

        st.divider()

        st.caption("Example 12")
        user12 = st.text_input(value="How would I style black t-shirt for a casual look?", label="user", key="user12")
        bot12 = st.text_input(
            value="We have Zara Mens Black T-Shirt, Levis Mens 501 Jeans, and Vans Old Skool Sneakers in stock. These are three classic pieces that can be dressed up or down. You can pair them with a jacket, hoodie, or cardigan.",
            label="bot", key="bot12")

        st.divider()

    # Initialize examples
    examples = [{"input": {"author": "user", "content": user1}, "output": {"author": "bot", "content": bot1}},
                    {"input": {"author": "user", "content": user2}, "output": {"author": "bot", "content": bot2}},
                    {"input": {"author": "user", "content": user3}, "output": {"author": "bot", "content": bot3}},
                    {"input": {"author": "user", "content": user4}, "output": {"author": "bot", "content": bot4}},
                    {"input": {"author": "user", "content": user5}, "output": {"author": "bot", "content": bot5}},
                    {"input": {"author": "user", "content": user6}, "output": {"author": "bot", "content": bot6}},
                    {"input": {"author": "user", "content": user7}, "output": {"author": "bot", "content": bot7}},
                    {"input": {"author": "user", "content": user8}, "output": {"author": "bot", "content": bot8}}]

    examples2 = [{"input": {"author": "user", "content": user9}, "output": {"author": "bot", "content": bot9}},
                    {"input": {"author": "user", "content": user10}, "output": {"author": "bot", "content": bot10}},
                    {"input": {"author": "user", "content": user11}, "output": {"author": "bot", "content": bot11}},
                    {"input": {"author": "user", "content": user12}, "output": {"author": "bot", "content": bot12}}]

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["author"]):
            st.markdown(message["content"])

    # Accept user input
    query = st.chat_input()
    vector_ids = []
    res = []
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"author": "user", "content": query.replace('\'','') })
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            if agent == "LLaMA-2":
                # full_response, res = backend_obj.llama_agent(query)
                full_response, res = "LLaMA-2 is currently not available.", []

            elif agent == "PaLM-2":
                full_response, res = backend_obj.palm_agent(query)

            elif agent == "Gemini-Pro":
                full_response, res = backend_obj.gemini_agent(query)

            if res:
                vector_ids = backend_obj.df[backend_obj.df[backend_obj.label].isin([_[backend_obj.label] for _ in res])].index.tolist()

            # for response in q_chat['predictions']:
            #     full_response += response['candidates'][0]['content']
            #     message_placeholder.markdown(full_response + "▌")

            # Display q&a response
            message_placeholder.markdown(full_response if full_response else "")

            # Display the vector search results
            if vector_ids:
                print(vector_ids)
                print(len(backend_obj.images))
                for i, col in enumerate(st.columns(len(vector_ids))):
                    vars()[f"col_{i}"] = col
                for i, t in enumerate(vector_ids):
                    col = vars()[f"col_{str(i)}"]
                    col.caption(backend_obj.df[backend_obj.label][t])
                    col.image(backend_obj.images[t], use_column_width="always")

        # Append q&a response
        st.session_state.messages.append({"author": "assistant", "content": full_response.replace('\'','')})