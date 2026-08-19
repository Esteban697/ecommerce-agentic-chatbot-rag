"""Data Ingestion and Vector Store Creation for Product Catalog
This module handles the ingestion of product catalog data from a JSON file, splits the data into manageable chunks,
creates embeddings using OpenAI's embedding model, and stores the embeddings in a Chroma vector store for efficient
retrieval during product searches. RAG (Retrieval-Augmented Generation) is used to enhance the chatbot's ability to provide
accurate and relevant product information based on user queries.
"""

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import Any
import json
from pathlib import Path
import html
import re

model_name = "text-embedding-3-small"  # OpenAI's embedding model
chroma_persist_directory = "./chroma_db"  # Directory to persist the Chroma vector store


def deduplicate_products_in_separate_json(file_path: str) -> tuple[int, list[int]]:
    """Deduplicate products in a JSON file and save the unique products into separate JSON files per site_id."""

    source_path = Path(file_path)

    with open(source_path, "r", encoding="utf-8") as f:
        raw_products: list[dict[str, Any]] = json.load(f)

    site_ids = sorted({row["site_id"] for row in raw_products})
    print("Site IDs found in the dataset:", site_ids)
    print("Opened JSON file and loaded products. Deduplicating now...")

    seen = set()
    products: list[dict[str, Any]] = []

    for row in raw_products:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            products.append(dict(row))

    print(f"Deduplicated products. {len(products)} unique products remain. Saving site-specific JSON files...")

    products_by_site: dict[int, list[dict[str, Any]]] = {}
    for row in products:
        site_id = row.get("site_id")
        if site_id is not None:
            products_by_site.setdefault(site_id, []).append(row)

    for site_id, site_products in products_by_site.items():
        site_file_path = source_path.with_name(
            f"{source_path.stem}_deduplicated_site_{site_id}.json"
        )
        with open(site_file_path, "w", encoding="utf-8") as f:
            json.dump(site_products, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(site_products)} deduplicated products for site_id={site_id} to {site_file_path}")

    return len(products), site_ids

def strip_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

def load_json_data(file_path: str) -> list[dict[str, Any]]:
    """Load product JSON data and convert each product into a searchable Document."""
    with open(file_path, "r", encoding="utf-8") as f:
        products: list[dict[str, Any]] = json.load(f)

    documents: list[dict[str, Any]] = []

    for i, row in enumerate(products):
        page_content = "\n".join([
            f"Product name: {row.get('product_name', '')}",
            f"Variant name: {row.get('variant_name', '')}",
            f"Brand: {row.get('brands', '')}",
            f"Pet type: {row.get('pet_type', '')}",
            f"Summary: {row.get('summary_text', '')}",
            f"Description: {row.get('description_text', '')}",
            f"Ingredients: {row.get('ingredients', '')}",
            f"Feeding recommendations: {row.get('feeding_recommendations', '')}",
            f"Search text: {row.get('search_blob', '')}",
        ]).strip()

        metadata = {
            "source": str(file_path),
            "seq_num": i,
            "site_id": row.get("site_id"),
            "product_id": row.get("product_id"),
            "article_id": row.get("article_id"),
            "product_name": row.get("product_name"),
            "variant_name": row.get("variant_name"),
            "brands": row.get("brands"),
            "pet_type": row.get("pet_type"),
        }

        documents.append({"page_content": page_content, "metadata": metadata})

    return documents

def embeddings_and_vectorstore(file_path: str | Path, site_id: int, max_chunk_size: int = 300) -> Chroma:
    """Load JSON data, split it into chunks, create embeddings, and store them in a vector store."""

    # Load the JSON data from the specified file path
    documents_data = load_json_data(str(file_path))

    # Directly convert dictionaries to Document objects to avoid RecursiveJsonSplitter
    # wrapping the page_content in JSON strings, which breaks extraction regex.
    documents_split = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in documents_data
    ]
    # Create embeddings using OpenAI's embedding model

    # Note: Requires OPENAI_API_KEY environment variable to be set
    embeddings = OpenAIEmbeddings(
        model=model_name,  # OpenAI's text-embedding-3-small model (1536 dimensions)
        # Other available models:
        # - "text-embedding-3-large" (3072 dimensions, higher quality)
        # - "text-embedding-ada-002" (1536 dimensions, legacy model)
    )
    persist_directory = Path(chroma_persist_directory) / f"site_{site_id}"
    collection_name = f"rag-chroma-site-{site_id}"

    # Create a Chroma vector store from the split documents and embeddings, and persist it to disk
    vectorstore = Chroma.from_documents(
        documents_split,
        embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    # Report what happened during the ingestion process
    print(f"Chroma vector store created with {len(documents_split)} documents.")
    print(f"Embedding model: {model_name}")
    print(f"Vector store persisted at: {chroma_persist_directory}")

# Provide a way to retrieve the vector store for use in other parts of the application
def get_vectorstore(site_id: int, chroma_base_dir: str | Path = "./chroma_db") -> Chroma:
    embeddings = OpenAIEmbeddings(model=model_name)
    persist_directory = Path(chroma_base_dir) / f"site_{site_id}"

    vectorstore = Chroma(
        collection_name=f"rag-chroma-site-{site_id}",
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    print(
        f"Loaded site-specific Chroma vector store for site_id={site_id} "
        f"from {persist_directory} with {vectorstore._collection.count()} documents."
    )
    return vectorstore