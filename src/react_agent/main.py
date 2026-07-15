""" Main entry point for the FastAPI backend.
Defines the API endpoints and the data ingestion logic."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from pathlib import Path

from typing import Any

from react_agent.context import Context
from react_agent.graph import graph
from react_agent.prompts import SYSTEM_PROMPT
from react_agent.ingestion import embeddings_and_vectorstore

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

@app.get('/')
async def root():
    return { 'message': 'API is running...'}

@app.get('/health')
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "sites_available": SITE_IDS,
        "raw_rows": len(RAW_PRODUCTS),
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
            model="openai/gpt-4.1",
            system_prompt=SYSTEM_PROMPT,
            site_id=payload.site_id,
            products=RAW_PRODUCTS,
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
Description: TODO
"""
# Make sure to load environment variables from .env file
load_dotenv()

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "react_agent" / "product_catalog_dataset.json"

# Initialize the embeddings and vector store for product retrieval
embeddings_and_vectorstore(DATA_PATH)

# THIS IS MOCK FROM HERE DOWN
with open(DATA_PATH, "r", encoding="utf-8") as f:
    RAW_PRODUCTS = json.load(f)

SITE_IDS = sorted({p["site_id"] for p in RAW_PRODUCTS})

print(f"Loaded {len(RAW_PRODUCTS)} products from {DATA_PATH}.")
print(f"Available site_ids: {SITE_IDS}")

