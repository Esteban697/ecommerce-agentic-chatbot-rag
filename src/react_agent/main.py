""" Main entry point for the FastAPI backend.
Defines the API endpoints and the data ingestion logic."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from pathlib import Path

from typing import Any

from react_agent.context import Context
from react_agent.graph import graph, check_model_active
from react_agent.prompts import SYSTEM_PROMPT
from react_agent.ingestion import embeddings_and_vectorstore, deduplicate_products_in_separate_json

from langchain_core.messages import HumanMessage

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


class ChatResponse(BaseModel):
    answer: str
    retrieved_products: list[dict[str, Any]]

model_name = "openai/gpt-4.1"  # Default model for the agent. Can be changed in the /chat endpoint if needed.

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

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content=payload.query)],
            "retrieved_products": [],
        },
        context = Context(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            site_id=payload.site_id,
            products=[], # TODO check if this makes sense
        ),
    )

    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    return ChatResponse(
        answer=answer,
        retrieved_products=result.get("retrieved_products", []),
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