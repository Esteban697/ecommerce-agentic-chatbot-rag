
### Dev Log: [not updated]

After reading the asignment I can split the focus in several aspects of this PoC:
- The API Endpoint needs to meet all criterias and work with the current structure of the data handled in the chat
- The RAG pipeline should make sense to operate accordingly
- Chatbot may be mocked and later it can be scaled to a real AI agent and then it's topology should be scalable without big changes and risks to the above mentioned

#### RAG Pipeline

and looking at the dataset provided I start with:

I analyzed the JSON dataset provided with Perpeplexity asking about:
- How many products in the dataset?
- How many site different site ids?
- Which field is better to describe each product separately: product_id or article_id?
- Are there fields that have the same values?
- Are there any outliers?

The first answer provided pointed at a 300 product list that was better identifying individual products
via article_id field and that duplicates were found.

These findings bring up several considerations for designing the solution's RAG pipeline:
- The ingestion stage starts by fetching local data JSON file
- Then, a pre-processing is desired for cleaning the data is useful
- Then, chunking needs to obey the data structure to keep the JSON relationships of the features of each product
- Then, embeddings should be simple and fast for the amount of products to handle in with the Chatbot
- Finally, the vector store may be a simple local vector DB that can be accessed in a tool node

Considering the API contract example that asks for a required "site_id" value we can create 3 separate vector DBs
following the steps described above to optimize the queries.

Based on this, technical decisions for this stage to prepare the data include:
Use langchain_community.document_loaders to fetch JSON file and start using the langchain_text_splitters 
(more espefically the RecursiveJsonSplitter) for the chunking of the JSON data rows.
These chunks will be used for embedding with LangChain's NomicEmbeddings integration, needing a NOMIC_API_KEY.
These embeddings will be indexed and store in a InMemoryVectorStore from langchain_core.vectorstores.
The resulting vector DB can be used as part of the logic of a Tool Node in the Chatbot's graph.
