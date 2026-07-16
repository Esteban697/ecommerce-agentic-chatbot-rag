"""Data Ingestion and Vector Store Creation for Product Catalog
This module handles the ingestion of product catalog data from a JSON file, splits the data into manageable chunks,
creates embeddings using Nomic's embedding model, and stores the embeddings in a Chroma vector store for efficient 
retrieval during product searches. RAG (Retrieval-Augmented Generation) is used to enhance the chatbot's ability to provide
accurate and relevant product information based on user queries.
"""

from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from typing import Any
import json
from pathlib import Path
import html
import re
import os
import nomic

model_name = "nomic-embed-text-v1.5"  # Nomic's embedding model
chroma_persist_directory = "./chroma_db"  # Directory to persist the Chroma vector store


def deduplicate_products_in_separate_json(file_path: str) -> tuple[int, list[int]]:
    """Deduplicate products in a JSON file and save the unique products to a new JSON file
    Args:
        file_path (str): Path to the input JSON file containing product data.
    Returns:
        int: Number of unique products after deduplication.
        list[int]: List of unique site IDs found in the product data.
    """
    
    RAW_PRODUCTS: list[dict[str, Any]] = json.load(open(file_path, "r", encoding="utf-8"))
    SITE_IDS = sorted({row['site_id'] for row in RAW_PRODUCTS})
    print("Site IDs found in the dataset:", SITE_IDS)
    print("Opened JSON file and loaded products. Deduplicating now...")
    # Deduplicate exact duplicates, which exist in the dataset for some article_ids.
    seen = set()
    PRODUCTS: list[dict[str, Any]] = []
    for row in RAW_PRODUCTS:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            row = dict(row)
            row['summary_text'] = strip_html(row.get('summary'))
            row['description_text'] = strip_html(row.get('description'))
            row['search_blob'] = ' '.join([
                str(row.get('product_name', '')),
                str(row.get('variant_name', '')),
                str(row.get('brands', '')),
                str(row.get('pet_type', '')),
                row['summary_text'],
                row['description_text'],
                str(row.get('ingredients', '')),
                str(row.get('feeding_recommendations', '')),
            ]).lower()
            PRODUCTS.append(row)
    print(f"Deduplicated products. {len(PRODUCTS)} unique products remain. Saving to new JSON file...")
    # Save the deduplicated products to a new JSON file
    deduplicated_file_path = Path(str(file_path).replace(".json", "_deduplicated.json"))
    with open(deduplicated_file_path, "w", encoding="utf-8") as f:
        json.dump(PRODUCTS, f)
    print(f"Saved deduplicated products to {deduplicated_file_path}")
    return len(PRODUCTS), SITE_IDS

def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def load_json_data(file_path: str) -> list[dict[str, Any]]:
    """Load JSON data from a file and return it as a list of dictionaries."""
    loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False,)
    documents = loader.load()
    return [doc.metadata for doc in documents]

def embeddings_and_vectorstore(file_path: str, max_chunk_size: int = 300) -> Chroma:
    """Load JSON data, split it into chunks, create embeddings, and store them in a vector store."""

    #log in to Nomic using the API key from environment variables
    nomic.login(token=os.environ["NOMIC_API_KEY"])
    #the data set will be split into JSON chunks of this size (in characters) before creating embeddings
    splitter = RecursiveJsonSplitter(max_chunk_size)
    #load the JSON data from the specified file path
    documents = load_json_data(file_path)
    #split the documents into smaller chunks for embedding
    documents_split = splitter.create_documents(documents)
    #create embeddings using Nomic's embedding model (extra param info taken from documentation)
    embeddings = NomicEmbeddings(
        model=model_name,
        # dimensionality=256,
        # Nomic's `nomic-embed-text-v1.5` model was [trained with Matryoshka learning](https://blog.nomic.ai/posts/nomic-embed-matryoshka)
        # to enable variable-length embeddings with a single model.
        # This means that you can specify the dimensionality of the embeddings at inference time.
        # The model supports dimensionality from 64 to 768.
        # inference_mode="remote",
        # One of `remote`, `local` (Embed4All), or `dynamic` (automatic). Defaults to `remote`.
        # api_key=... , # if using remote inference,
        # device="cpu",
        # The device to use for local embeddings. Choices include
        # `cpu`, `gpu`, `nvidia`, `amd`, or a specific device name. See
        # the docstring for `GPT4All.__init__` for more info. Typically
        # defaults to CPU. Do not use on macOS.
    )
    #create a Chroma vector store from the split documents and embeddings, and persist it to disk
    vectorstore = Chroma.from_documents(
        documents_split,
        embeddings,
        collection_name="rag-chroma",
        persist_directory=chroma_persist_directory
    )
    #Report what happened during the ingestion process
    print(f"Chroma vector store created with {len(documents_split)} documents.")
    print(f"Embedding model: {embeddings.model}, Dimensionality: {embeddings.dimensionality}")
    print(f"Vector store persisted at: {chroma_persist_directory}")

# Provide a way to retrieve the vector store for use in other parts of the application
def get_vectorstore() -> Chroma:
    """Load the persisted vector store from disk."""
    embeddings = NomicEmbeddings(model=model_name, inference_mode="local")
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding=embeddings,
        persist_directory=chroma_persist_directory
    )
    return vectorstore