import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, SupabaseVectorStore, Qdrant
from typing import List
from langchain.docstore.document import Document
from qdrant_client import QdrantClient

from modules.MyNotionDBLoader import MyNotionDBLoader


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
    """
    page_content = doc.page_content \
        .replace("\ue05c", "fi") \
        .replace("\ufb01", "fi") \
        .replace("\x00", "") \
        .replace('\u0000', '')
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
        # https://qdrant.tech/documentation/how_to/#prefer-high-precision-with-high-speed-search
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


def load_qdrant(args):
    load_dotenv()

    docs = load_notion_documents(notion_token=os.getenv("NOTION_TOKEN"),
                                 notion_database_id=os.getenv("NOTION_DATABASE_ID"))
    doc_chunks = split_documents(docs)

    q_client = get_qdrant_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

    if args.reset:
        print("\nStart recreating Qdrant collection...")
        recreate_collection(q_client)
        print("Finished recreating Qdrant collection")

    if args.verbose:
        collection_info = q_client.get_collection(collection_name=os.getenv("QDRANT_COLLECTION_NAME"))
        print(f"\nCollection info: {collection_info.json()}")

    vectors = Qdrant(
        client=q_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding_function=OpenAIEmbeddings().embed_query,
    )

    print("\nStart loading documents to Qdrant...")

    CHUNK_SIZE = 50
    print(f"Number of documents: {len(doc_chunks)}")
    doc_chunks_list = [doc_chunks[i:i + CHUNK_SIZE] for i in range(0, len(doc_chunks), CHUNK_SIZE)]
    print(f"Number of batches: {len(doc_chunks_list)}")

    for j in range(0, len(doc_chunks_list)):
        print(f"Loading batch number {j + 1}...")

        Qdrant.add_documents(
            self=vectors,
            documents=doc_chunks_list[j],
        )

    print("Finished loading documents to Qdrant")

    if args.verbose:
        collection_info = q_client.get_collection(collection_name=os.getenv("QDRANT_COLLECTION_NAME"))
        print(f"\nCollection info: {collection_info.json()}")


def qdrant_test():
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

