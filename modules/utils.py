import os
import pandas as pd
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from modules.chatbot import Chatbot

from modules.qdrant_util import get_qdrant_client


class Utilities:
    @staticmethod
    def load_api_key():
        """
        Loads the OpenAI API key from the .env file or from the user's input
        and returns it
        """
        if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
            user_api_key = os.environ["OPENAI_API_KEY"]
            st.sidebar.success("API key loaded", icon="🚀")
        # elif st.secrets["OPENAI_API_KEY"]:
        #     user_api_key = st.secrets["OPENAI_API_KEY"]
        else:
            user_api_key = st.sidebar.text_input(
                label="#### Your OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password"
            )
            if user_api_key:
                st.sidebar.success("API key loaded", icon="🚀")
        return user_api_key

    @staticmethod
    def handle_upload():
        """
        Handles the file upload and displays the uploaded file
        """
        uploaded_file = st.sidebar.file_uploader("upload", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:
            

            def show_user_file(uploaded_file):
                file_container = st.expander("Your CSV file :")
                shows = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                file_container.write(shows)
             

            show_user_file(uploaded_file)
        else:
            st.sidebar.info(
                "👆 Upload your CSV file to get started, "
                "sample for try : [fishfry-locations.csv](https://drive.google.com/file/d/1TpP3thVnTcDO1_lGSh99EKH2iF3GDE7_/view?usp=sharing)"
            )
            st.session_state["reset_chat"] = True
        return uploaded_file

    @staticmethod
    def setup_chatbot(uploaded_file, model, temperature):
        """
        Sets up the chatbot with the uploaded file, model, and temperature
        """
        with st.spinner("Processing..."):
            # uploaded_file.seek(0)
            # file = uploaded_file.read()


            # vectors = faiss_fetch_vector_store()
            q_client = get_qdrant_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

            vectors = Qdrant(
                client=q_client,
                collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                embedding_function=OpenAIEmbeddings().embed_query,
            )

            # vectors = embeds.getDocEmbeds(file, uploaded_file.name)
            chatbot = Chatbot(model, temperature, vectors)
        return chatbot
