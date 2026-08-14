#!/usr/bin/env python3
"""Standalone Data Ingestion Script for Product Catalog

This script handles the complete data ingestion process:
1. Deduplicates products from the JSON dataset
2. Creates embeddings using Nomic's embedding model
3. Stores embeddings in a Chroma vector store for efficient retrieval

Usage:
    python -m react_agent.ingest
    python -m react_agent.ingest --data-path path/to/data.json
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
import json

from react_agent.ingestion import (
    embeddings_and_vectorstore,
    deduplicate_products_in_separate_json,
)


def main():
    """Run the complete data ingestion pipeline."""

    parser = argparse.ArgumentParser(
        description="Ingest product catalog data and create vector stores"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the product catalog JSON file (default: product_catalog_dataset.json in react_agent dir)"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=15000,
        help="Maximum chunk size for document splitting (default: 15000)"
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")

    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        app_dir = Path(__file__).resolve().parent
        data_path = app_dir / "product_catalog_dataset.json"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return 1

    print(f"\n{'='*60}")
    print("Starting Data Ingestion Process")
    print(f"{'='*60}\n")
    print(f"Data source: {data_path}")

    # Step 1: Deduplicate products and create site-specific JSON files
    print(f"\n{'='*60}")
    print("Step 1: Deduplicating products...")
    print(f"{'='*60}\n")

    unique_products_count, site_ids = deduplicate_products_in_separate_json(data_path)

    print(f"\nDeduplication complete!")
    print(f"  - Unique products: {unique_products_count}")
    print(f"  - Available site IDs: {site_ids}")

    # Step 2: Create embeddings and vector stores for each site
    print(f"\n{'='*60}")
    print("Step 2: Creating embeddings and vector stores...")
    print(f"{'='*60}\n")

    for site_id in site_ids:
        site_json_path = data_path.with_name(
            f"{data_path.stem}_deduplicated_site_{site_id}.json"
        )
        print(f"\nProcessing site_id={site_id}")
        print(f"  - Source: {site_json_path.name}")
        print(f"  - Max chunk size: {args.max_chunk_size}")

        embeddings_and_vectorstore(
            site_json_path,
            site_id=site_id,
            max_chunk_size=args.max_chunk_size
        )

    print(f"\n{'='*60}")
    print("Data Ingestion Complete!")
    print(f"{'='*60}\n")
    print(f"✓ Processed {unique_products_count} unique products")
    print(f"✓ Created vector stores for {len(site_ids)} sites: {site_ids}")
    print(f"✓ Vector stores saved to ./chroma_db/")
    print(f"\nYou can now run the FastAPI server with:")
    print(f"  uvicorn react_agent.main:app --reload")

    return 0


if __name__ == "__main__":
    exit(main())
