## Dev Log:

After reading the asignment I can split the focus in several aspects of this PoC:
- The API Endpoint needs to meet all criterias and work with the current structure of the data handled in the chat
- The RAG pipeline should make sense to operate accordingly
- Chatbot may be mocked and later it can be scaled to a real AI agent and then it's topology should be scalable without big changes and risks to the above mentioned

### 1) API Endpoint:

AI-assisted coding was able to put together an initial template of what the FastAPI Post method could look like.
Also, provided relevant code to de-duplicate the article-products in the dataset and a Mock to provide answers from that data when called.
Nevertheless, I added to the current structure of the React Agent template from LangChain to connect it to a relevant component that can act as ChatBot.

#### GET root (/):
Just allows for a quick check when running the Uvicorn server locally, you can see the confirmation message in the browser when it loads the root.

#### GET (/health)
I useful call to check that API is running, that model is being able to respond to the calls, which site IDs
can be used for the chatbot to have information about and that ingestion stagewas able to sort the products
in the dataset provided.

#### POST (/chat)
The actual exposed method to use the chatbot from any compatible application. It expects a specific JSON:

[JSON]

It accessed the graph built with LangGraph to be able to activate the agent that can process what is asked as 
Human message and use its tool (mainly RAG tool) to inspect if a product/s can match the request.
This process expects the agent to analyze the retrieve results from the dataset and produce an polite and useful
answer as it will be the expected string to be displayed in the chat for the end user.

The response has the following JSON structure:

[JSON]

### 2) RAG Pipeline

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

### 3) Chatbot

API can provide a way to test the Chatbot implemented and exposed the data structures required.
LangGraph based on ToolNode providing calls to RAG Pipeline and LangSmith providing observability.
----------------------------------
# Delivery Structure:

## High-Level Design
A brief overview of your solution's architecture. A simple diagram is welcome.

## Setup and Execution
Clear instructions to set up the environment and run the project locally.

## Decisions and Trade-offs
Explain the key decisions you made in designing your agent and retrieval pipeline. What trade-offs did you consciously accept for the sake of this PoC, and what would you change or recommend for a production deployment?

## Future Roadmap
If this PoC were approved, what would be the next 3-5 steps to move it towards production? (Think about testing, observability, evaluation, scalability, additional features, etc.)
-----------------------------------