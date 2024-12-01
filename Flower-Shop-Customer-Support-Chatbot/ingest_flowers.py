from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

import config
import pandas as pd
import re


def ingest_flowers_to_qdrant():
    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    print("Load flowers data Start")

    df = pd.read_excel("dataset/flowers.xlsx")

    print("Load flowers data end")

    # --------------------------------------------------------
    # 2. Prepare Chunks
    # --------------------------------------------------------
    print("Prepare Chunks Start")
    langchain_documents = []
    for index, row in df.iterrows():
        product_id = row['Product_Id']
        product_name = row['Product_Name']
        best_occasion = row['Best_Occasion']
        description = row['Description']
        price_rm = row['Price_RM']
        blooms = row['Blooms']
        bouquet_size = row['Bouquet_Size']

        content = f"""Product Name: {product_name}
    Best Occasion: {best_occasion}
    Product Description: {description}
        """

        document = Document(
            page_content=content,
            metadata={
                "product_id":product_id,
                "product_name":product_name,
                "price_rm":price_rm,
                "blooms":blooms,
                "bouquet_size":bouquet_size
            }
        )

        langchain_documents.append(document)

    print("Prepare Chunks End")

    # --------------------------------------------------------
    # 3. Ingest to Qdrant
    # --------------------------------------------------------
    print("Ingest to Vector Database Start.")
    url = config.QDRANT_URL
    api_key = config.QDRANT_API_KEY

    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )

    model_name = "jinaai/jina-embeddings-v3"
    model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': False}
    jina_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    embeddings = jina_embeddings

    QdrantVectorStore.from_documents(
        langchain_documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name="flowers",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    print("Ingest to Vector Database End.")

def ingest_policy_to_qdrant():
    # ---------------------------------------------------
    # 1. Load Documents
    # ---------------------------------------------------
    print("Loading policy documents Start.")
    file_path = "Online_Flower_Shop_Policies.pdf"
    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(file_path)  # Replace with the path to your PDF file
    documents = loader.load()
    print("Loading policy documents End.")

    # ---------------------------------------------------
    # 2. Prepare Chunks
    # ---------------------------------------------------
    print("Preparing Chunks Start.")
    text = ""
    for document in documents:
        text += document.page_content

    # Use regex to split the text based on numbering (works for single and double digit numbers)
    sections = re.split(r'(?=\d{1,2}\.\s)', text.strip())  # Matches 1- or 2-digit numbers followed by '. '

    # Remove empty strings and strip whitespace
    sections = [section.strip() for section in sections if section.strip()]

    langchain_documents = [Document(page_content=section, metadata={'document_type':'policy'}) for section in sections]
    print("Preparing Chunks End.")

    # ---------------------------------------------------
    # 3. Ingest to Qdrant
    # ---------------------------------------------------
    print("Ingest to Vector Database Start.")
    url = config.QDRANT_URL
    api_key = config.QDRANT_API_KEY

    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )

    model_name = "jinaai/jina-embeddings-v3"
    model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': False}
    jina_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    embeddings = jina_embeddings

    QdrantVectorStore.from_documents(
        langchain_documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=url,
        prefer_grpc=True,
        api_key=api_key,
        collection_name="flowers_policy",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    print("Ingest to Vector Database End.")



if __name__ == "__main__":
    ingest_flowers_to_qdrant()
    ingest_policy_to_qdrant()













