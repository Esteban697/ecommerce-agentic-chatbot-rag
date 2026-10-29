""" Main entry point for the FastAPI backend.
Defines the API endpoints and the data ingestion logic."""

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
from uuid import uuid4

import json
import time
from pathlib import Path
from typing import Any

def clean_llm_answer(text: str) -> str:
    """
    Aggressively cleans the LLM response to ensure it's a plain string.
    """
    if not text:
        return ""

    current_text = text.strip()

    # Strip markdown code blocks
    if current_text.startswith("```"):
        import re
        current_text = re.sub(r"^```[a-zA-Z]*\n?", "", current_text)
        current_text = re.sub(r"\n?```$", "", current_text)
        current_text = current_text.strip()

    # We loop to handle multiple layers of wrapping
    for _ in range(5):
        # 1. Try JSON parsing
        try:
            data = json.loads(current_text)
            if isinstance(data, dict):
                if "answer" in data:
                    current_text = str(data["answer"]).strip()
                    continue
                if len(data) == 1:
                    current_text = str(list(data.values())[0]).strip()
                    continue
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Regex fallback: find the value of the 'answer' key
        import re
        # This regex looks for "answer": "..." and captures the content
        match = re.search(r'"answer":\s*"(.*?)"\s*\}?$', current_text, re.DOTALL | re.IGNORECASE)
        if match:
            current_text = match.group(1).strip()
            continue

        # 3. Another regex fallback for cases where it's just { "answer": "..." }
        match = re.search(r'\{\s*"answer":\s*"(.*?)"\s*\}', current_text, re.DOTALL | re.IGNORECASE)
        if match:
            current_text = match.group(1).strip()
            continue

        break

    # Final pass: if it still looks like a JSON object with an 'answer' key,
    # but regex/json failed, try a very simple split.
    if current_text.startswith('{') and '"answer":' in current_text:
        try:
            # Very crude extraction
            parts = current_text.split('"answer":')
            if len(parts) > 1:
                val = parts[1].strip()
                if val.startswith('"'):
                    val = val[1:]
                if val.endswith('"') or val.endswith('"}'):
                    val = val.rstrip('"}')
                return val.strip()
        except Exception:
            pass

    return current_text

from react_agent.context import Context
from react_agent.graph import graph, check_model_active
from react_agent.prompts import SYSTEM_PROMPT

from langchain_core.messages import HumanMessage, BaseMessage


""" API Endpoint definition with FastAPI
Description: Defines the API endpoints for the E-Commerce Chatbot Service.
"""

app = FastAPI(
    title="E-Commerce Chatbot Service",
    version="1.0.0",
    description="Async FastAPI backend that answers catalog questions using site-scoped product retrieval."
)

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
async def root():
    return FileResponse("static/index.html")

# API v1 Router for versioned endpoints
v1_router = APIRouter(prefix="/v1", tags=["v1"])

class ChatRequest(BaseModel):
    site_id: int = Field(..., description="Shop context identifier")
    query: str = Field(..., min_length=1, max_length=2000, description="Customer natural language question")
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
    # Internal commercial metrics (stock_units, margin_pct, monthly_sales_units,
    # revenue_last_30d) are deliberately NOT part of the public response model.

class ChatResponse(BaseModel):
    answer: str
    retrieved_products: list[Product]
    conversation_id: str

model_name = "openai/gpt-4.1"

# `/health` used to invoke the model on every request, so a loop against it
# burned provider quota. Cache the probe for a short window instead.
_HEALTH_TTL_SECONDS = 300
_health_checked_at: float | None = None
_health_is_active: bool = False


async def cached_model_active(model: str) -> bool:
    """Return the cached model-health result, refreshing at most every TTL."""
    global _health_checked_at, _health_is_active
    now = time.monotonic()
    if _health_checked_at is not None and now - _health_checked_at < _HEALTH_TTL_SECONDS:
        return _health_is_active
    _health_is_active = await check_model_active(model)
    _health_checked_at = now
    return _health_is_active

# Simple in-memory conversation store
CONVERSATIONS: dict[str, list[BaseMessage]] = {}
# Maximum number of messages to keep in memory for a conversation.
# Longer conversations should be saved externally and reloaded as needed.
MAX_MESSAGES = 30
# Cap on concurrently retained conversations. Dicts preserve insertion order,
# so the first key is the oldest and is evicted first.
MAX_CONVERSATIONS = 1000
# Simple in-memory ids store
# ID_HISTORY is no longer used globally to ensure thread-safety and session isolation.

@v1_router.get('/health')
async def health() -> dict[str, Any]:
    """Health check endpoint for API and model status."""
    is_model_active = await cached_model_active(model_name)
    return {
        "API_status": "Ok",
        "model_active": is_model_active,
        "sites_available": SITE_IDS,
        "unique_products": UNIQUE_PRODUCTS_COUNT,
        }


@v1_router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Main chat endpoint for conversational product recommendations."""
    if payload.site_id not in SITE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown site_id {payload.site_id}. Available site_ids: {SITE_IDS}",
        )

    conversation_id = payload.conversation_id or str(uuid4())
    history = CONVERSATIONS.get(conversation_id, [])[-MAX_MESSAGES:]
    # ID_HISTORY removed to prevent shadowing and ensure session isolation.

    # Update the context with the current site_id and products for this conversation
    current_user_message = HumanMessage(content=payload.query)
    input_messages = history + [current_user_message]
    # Retrieve previously retrieved products for this conversation, if any
    previous_products = [Product(**p) for p in CONVERSATIONS.get(conversation_id, []) if isinstance(p, Product)]

    result = await graph.ainvoke(
        {
            "messages": input_messages,
            "retrieved_products": previous_products,
            "retrieved_products_ids": [],
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

    # Clean the answer to remove potential double-encoded JSON
    answer = clean_llm_answer(answer)

    updated_messages = result["messages"]
    if conversation_id not in CONVERSATIONS and len(CONVERSATIONS) >= MAX_CONVERSATIONS:
        # Evict the oldest conversation; this store is process-local and bounded.
        CONVERSATIONS.pop(next(iter(CONVERSATIONS)), None)
    CONVERSATIONS[conversation_id] = updated_messages

    # Get the list of IDs from graph State, originally from tool call
    retrieved_products_ids = result.get("retrieved_products_ids", [])
    print(f"retrieved_products_ids from graph result: {retrieved_products_ids}")
    # ID_HISTORY append removed for session isolation.

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

# Include the v1 router in the main app
app.include_router(v1_router)

print("Starting FastAPI backend for E-Commerce Chatbot Service...")

""" Data Loading
Description: Load pre-processed product data from deduplicated JSON files.
If the deduplicated files don't exist, run the ingestion script first:
    python -m react_agent.ingest
"""
# Make sure to load environment variables from .env file
load_dotenv()

# Variables that can be inspected in the FastAPI /health endpoint
SITE_IDS = []
UNIQUE_PRODUCTS_COUNT = 0
RAW_PRODUCTS = []

# Load data from deduplicated files
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "product_catalog_dataset.json"

# Check if deduplicated files exist
deduplicated_files = list(APP_DIR.glob("product_catalog_dataset_deduplicated_site_*.json"))

if not deduplicated_files:
    print("\n" + "="*60)
    print("WARNING: No deduplicated data files found!")
    print("="*60)
    print("Please run the ingestion script first:")
    print("  python -m react_agent.ingest")
    print("="*60 + "\n")
    raise FileNotFoundError(
        "Deduplicated data files not found. Run 'python -m react_agent.ingest' first."
    )

# Load the original product catalog for the full dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    RAW_PRODUCTS = json.load(f)

# Extract site IDs from deduplicated filenames
SITE_IDS = sorted([
    int(f.stem.split("_site_")[-1])
    for f in deduplicated_files
])

# Count unique products from the original dataset
seen_keys = set()
for row in RAW_PRODUCTS:
    key = json.dumps(row, sort_keys=True, ensure_ascii=False)
    seen_keys.add(key)
UNIQUE_PRODUCTS_COUNT = len(seen_keys)

print(f"Loaded {len(RAW_PRODUCTS)} products (including duplicates) from {DATA_PATH}.")
print(f"Unique products: {UNIQUE_PRODUCTS_COUNT}")
print(f"Available site_ids: {SITE_IDS}")
print(f"Vector stores should be available in ./chroma_db/ for sites: {SITE_IDS}")