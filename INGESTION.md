# Data Ingestion Guide

This guide explains how to run the data ingestion process separately from the FastAPI server.

## Overview

The data ingestion process has been separated into a standalone script that you only need to run once (or whenever you update the product catalog). The server will load from the pre-processed data files.

## Running Data Ingestion

### First Time Setup

1. Make sure you have your `.env` file configured with the required API keys:
   ```bash
   NOMIC_API_KEY=your_nomic_api_key_here
   ```

2. Run the ingestion script:
   ```bash
   python -m react_agent.ingest
   ```

This will:
- Deduplicate products from `product_catalog_dataset.json`
- Create site-specific JSON files (e.g., `product_catalog_dataset_deduplicated_site_1.json`)
- Generate embeddings using Nomic's embedding model
- Store embeddings in `./chroma_db/` directory for each site

### Custom Data Path

If your product catalog is in a different location:

```bash
python -m react_agent.ingest --data-path path/to/your/data.json
```

### Custom Chunk Size

To adjust the maximum chunk size for document splitting (default is 15000):

```bash
python -m react_agent.ingest --max-chunk-size 10000
```

## Running the FastAPI Server

After ingestion is complete, start the server without re-running ingestion:

```bash
uvicorn react_agent.main:app --reload
```

The server will:
- Load product data from the deduplicated files
- Use the pre-built vector stores from `./chroma_db/`
- Start much faster since it skips the ingestion process

## When to Re-run Ingestion

You only need to run the ingestion script again when:
- The product catalog data changes
- You want to change the chunk size
- The vector store gets corrupted or deleted

## Troubleshooting

### "No deduplicated data files found"

If you see this error when starting the server:
```
WARNING: No deduplicated data files found!
Please run the ingestion script first:
  python -m react_agent.ingest
```

Simply run the ingestion script as shown above.

### Missing Environment Variables

If you get an error about missing `NOMIC_API_KEY`:
1. Create a `.env` file in the root directory
2. Add your Nomic API key: `NOMIC_API_KEY=your_key_here`
3. Run the ingestion script again

## File Structure

After ingestion, you should see:
```
src/react_agent/
├── product_catalog_dataset.json                    # Original data
├── product_catalog_dataset_deduplicated_site_1.json  # Deduplicated per site
├── product_catalog_dataset_deduplicated_site_3.json
├── product_catalog_dataset_deduplicated_site_15.json
└── ...

chroma_db/
├── site_1/     # Vector store for site 1
├── site_3/     # Vector store for site 3
├── site_15/    # Vector store for site 15
└── ...
```
