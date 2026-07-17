""" Main entry point for the FastAPI backend.
Defines the API endpoints and the data ingestion logic."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
from uuid import uuid4

import json

from react_agent.context import Context
from react_agent.graph import graph, check_model_active
from react_agent.prompts import SYSTEM_PROMPT
from react_agent.ingestion import (
    embeddings_and_vectorstore,
    deduplicate_products_in_separate_json,
)

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


""" API Endpoint definition with FastAPI
Description: Defines the API endpoints for the E-Commerce Chatbot Service.
"""

app = FastAPI(
    title="E-Commerce Chatbot Service",
    version="1.0.0",
    description="Async FastAPI backend that answers catalog questions using site-scoped product retrieval."
)

class ChatRequest(BaseModel):
    site_id: int = Field(..., description="Shop context identifier")
    query: str = Field(..., min_length=1, description="Customer natural language question")
    conversation_id: str | None = Field(
        default=None,
        description="Conversation identifier. Reuse it across turns to preserve memory."
    )

class Product(BaseModel):
    product_id: int | None = None
    article_id: int | None = None
    variant_id: str | None = None
    site_id: int | None = None
    locale: str | None = None
    pet_type: str | None = None
    brands: str | None = None
    product_name: str | None = None
    variant_name: str | None = None
    summary: str | None = None
    description: str | None = None
    ingredients: str | None = None
    feeding_recommendations: str | None = None
    price: float | None = None
    currency: str | None = None
    discount_label: str | None = None
    rating_average: float | None = None
    rating_count: int | None = None
    stock_units: int | None = None
    margin_pct: float | None = None
    monthly_sales_units: int | None = None
    revenue_last_30d: float | None = None

class ChatResponse(BaseModel):
    answer: str
    retrieved_products: list[Product]
    conversation_id: str

model_name = "openai/gpt-4.1"

# Simple in-memory conversation store
CONVERSATIONS: dict[str, list[BaseMessage]] = {}
# Maximum number of messages to keep in memory for a conversation. 
# Longer conversations should be saved externally and reloaded as needed.
MAX_MESSAGES = 30
# Simple in-memory ids store
ID_HISTORY: list[int]

@app.get('/')
async def root():
    return { 'message': 'API is running...'}

@app.get('/health')
async def health() -> dict[str, Any]:
    is_model_active = await check_model_active(model_name)
    return {
        "API_status": "Ok",
        "model_active": is_model_active,
        "sites_available": SITE_IDS,
        "unique_products": UNIQUE_PRODUCTS_COUNT,
        }


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    if payload.site_id not in SITE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown site_id {payload.site_id}. Available site_ids: {SITE_IDS}",
        )

    conversation_id = payload.conversation_id or str(uuid4())
    history = CONVERSATIONS.get(conversation_id, [])[-MAX_MESSAGES:]
    ID_HISTORY = []

    # Update the context with the current site_id and products for this conversation
    current_user_message = HumanMessage(content=payload.query)
    input_messages = history + [current_user_message]
    # Retrieve previously retrieved products for this conversation, if any
    previous_products = [Product(**p) for p in CONVERSATIONS.get(conversation_id, []) if isinstance(p, Product)]

    result = await graph.ainvoke(
        {
            "messages": input_messages,
            "retrieved_products": previous_products,
            "retrieved_products_ids": ID_HISTORY
        },
        context=Context(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            site_id=payload.site_id,
            products=RAW_PRODUCTS,
        ),
    )

    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    updated_messages = result["messages"]
    CONVERSATIONS[conversation_id] = updated_messages

    # Get the list of IDs from graph State, originally from tool call
    retrieved_products_ids = result.get("retrieved_products_ids", [])
    print(f"retrieved_products_ids from graph result: {retrieved_products_ids}")
    ID_HISTORY.append(retrieved_products_ids)

    site_products = [
        p for p in RAW_PRODUCTS
        if p.get("site_id") == payload.site_id
    ]

    retrieved_products = [
        Product(**p)
        for p in site_products
        if p.get("article_id") in retrieved_products_ids
    ]

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        retrieved_products=retrieved_products
    )

print("Starting FastAPI backend for E-Commerce Chatbot Service...")

""" Data Ingestion 
Description: The ingestion stage for the JSON dataset goes as follows:
1. Deduplicate the products in the JSON dataset and save the unique products to a new JSON file.
2. Create embeddings for the unique products using Nomic's embedding model using the new JSON file.
3. Store the embeddings in a Chroma local vector store for efficient retrieval during product searches.
"""
# Make sure to load environment variables from .env file
load_dotenv()

# Variables that can be inspected in the FastAPI /health endpoint
SITE_IDS = []
UNIQUE_PRODUCTS_COUNT = 0

# Now running the ingestion process
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "react_agent" / "product_catalog_dataset.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    RAW_PRODUCTS = json.load(f)

UNIQUE_PRODUCTS_COUNT, SITE_IDS = deduplicate_products_in_separate_json(DATA_PATH)

# Initialize the embeddings and vector store for product retrieval
embeddings_and_vectorstore(Path(str(DATA_PATH).replace(".json", "_deduplicated.json")), 15000)  # Chunk size is based in JSON dataset provided

# MOCK--------------------------------------------------------------
# with open(DATA_PATH, "r", encoding="utf-8") as f:
#     RAW_PRODUCTS = json.load(f)

# SITE_IDS = sorted({p["site_id"] for p in RAW_PRODUCTS})
#-------------------------------------------------------------------

print(f"Loaded {UNIQUE_PRODUCTS_COUNT} products from {DATA_PATH}.")
print(f"Available site_ids: {SITE_IDS}.")