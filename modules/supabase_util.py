import os
import re

from supabase import create_client, Client
from dotenv import load_dotenv
import shutil

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, SupabaseVectorStore
from typing import List
from langchain.docstore.document import Document
from modules.MyNotionDBLoader import MyNotionDBLoader
from pathlib import Path


def load_notion_documents(notion_token, notion_database_id) -> List[Document]:
    loader = MyNotionDBLoader(notion_token, notion_database_id)
    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents from Notion")
    # if len(documents) > 0:
    #     print(f"\nFirst document: {documents[0]}")
    return documents


def split_documents(documents) -> List[Document]:
    clean_documents = [replace_non_ascii(doc) for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1024)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
        length_function=len,
    )
    document_chunks = text_splitter.split_documents(clean_documents)
    return document_chunks


def replace_non_ascii(doc: Document) -> Document:
    """
    Replaces non-ascii characters with ascii characters
    str = "This is Python \u200ctutorial"
    str_en = str.encode("ascii", "ignore")
    str_de = str_en.decode()
    print(str_de)
    """
    # page_content = doc.page_content.replace("\ue05c", "fi").replace("\x00", "")
    # page_content = doc.page_content.replace('\u0000', '')
    page_content = doc.page_content
    page_content_ascii = page_content.encode("ascii", "ignore").decode()
    output = ''.join([i if ord(i) < 128 else ' ' for i in page_content_ascii])
    # return Document(page_content="".join([i if ord(i) < 128 else " " for i in page_content]), metadata=doc.metadata)
    return Document(page_content=output, metadata=doc.metadata)


if __name__ == '__main__':
    load_dotenv()

    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    NOTION_TOKEN = os.getenv("NOTION_TOKEN")
    NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
    docs = load_notion_documents(notion_token=NOTION_TOKEN, notion_database_id=NOTION_DATABASE_ID)
    doc_chunks = split_documents(docs)

    supabase_vector_store = SupabaseVectorStore.from_documents(
        docs, OpenAIEmbeddings(), client=supabase
    )
