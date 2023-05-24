from qdrant_client import QdrantClient
import os

from dotenv import load_dotenv
import shutil

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, SupabaseVectorStore, Qdrant
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
    page_content = doc.page_content.replace("\ue05c", "fi").replace("\x00", "").replace('\u0000', '')
    page_content_ascii = page_content.encode("ascii", "ignore").decode()
    # output = ''.join([i if ord(i) < 128 else ' ' for i in page_content_ascii])
    return Document(page_content=page_content_ascii, metadata=doc.metadata)


def get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    qclient = QdrantClient(
        url=url,
        prefer_grpc=True,
        api_key=api_key,
    )
    return qclient


def recreate_collection(q_client: QdrantClient) -> None:
    from qdrant_client.http import models

    q_client.recreate_collection(
        collection_name="hybris",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )


# if __name__ == '__main__':
#     load_dotenv()
#
#     NOTION_TOKEN = os.getenv("NOTION_TOKEN")
#     NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
#     docs = load_notion_documents(notion_token=NOTION_TOKEN, notion_database_id=NOTION_DATABASE_ID)
#     doc_chunks = split_documents(docs)
#
#     q_client = get_qdrant_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))
#     recreate_collection(q_client)
#
#     # collection_info = q_client.get_collection(collection_name=os.getenv("QDRANT_COLLECTION_NAME"))
#     # print(f"\nCollection info: {collection_info}")
#
#     print("\nStart loading documents to Qdrant...")
#     qdrant = Qdrant.from_documents(
#         documents=doc_chunks,
#         embedding=OpenAIEmbeddings(),
#         url=os.getenv("QDRANT_URL"),
#         prefer_grpc=True,
#         api_key=os.getenv("QDRANT_API_KEY"),
#         collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
#     )
#     print("Finished loading documents to Qdrant")
#
#     # test
#     query = "What is Intelligent Selling Services?"
#     found_docs = qdrant.similarity_search_with_score(query)
#     found_doc, score = found_docs[0]
#     print(found_doc)
#     print(f"\nScore: {score}")
#
#     qdrant.similarity_search_with_score(query, filter={"source id": "hybrismart"})
#     found_doc, score = found_docs[0]
#     print(found_doc)
#     print(f"\nScore: {score}")

if __name__ == '__main__':
    load_dotenv()

    q_client = get_qdrant_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

    qdrant = Qdrant(
        client=q_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding_function=OpenAIEmbeddings().embed_query,
    )

    # test
    query = "What is Intelligent Selling Services?"
    found_docs = qdrant.similarity_search_with_score(query)
    print(f"\nFound {len(found_docs)} documents")
    found_doc, score = found_docs[0]
    print(found_doc)
    print(f"Score: {score}")

    found_docs = qdrant.similarity_search_with_score(query, filter={"source id": "help.ibm.com"})
    print(f"\n\nFound {len(found_docs)} documents")
    found_doc, score = found_docs[0]
    print(found_doc)
    print(f"Score: {score}")
