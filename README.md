## High-Level Design

The following diagram shows both how the different components connect to each other (without going inside of the LangGraph based graph that builds the agent) and also shows the expected sequence of components that run when using the Uvicorn server to run the PoC:

First, the execution takes us to the data ingestion stage that takes the JSON file with the product catalog that the diagram represents as Dataset 0. This will be parsed and get rid of duplicated products (detected in our first analysis of the dataset). After algorithmic cleaning and following step will produce one new JSON file per each site_id provided inside Dataset 0. This allows for a cleaner data management without the concern of mixing products between sites. Next, each JSON file will be treated as the source for embeddings and vector database to produce. The Nomic model persforms a remote inference to produce the embeddings and these are passed on to Chroma to produce a dedicated vector database for that group of products of a specific site. This will be useful when retrieval occurs.

Second, when the data is locally saved and prepared for consumption, the user can start sending queries through the API implemented with FastAPI. The POST call needs to provide a JSON with query and site_id (and optional conversation_id normally from second message onwards). If you provide the conversation_id that comes in the first response of this API call, you can enjoy a conversation the chatbot will follow with semantic memory.

Third, when the Chatbot agent receives queries, the LLM (in this case "openai/gpt-4.1") will decide when and which tool to call based on the system prompt that describes in detail what are the guardrails set for this chatbot to focus on product recommendation. The web_search tool based on Tavily is provided as a means to keep the chatbot situationally aware (even though a current_time value is passed to the system prompt) to avoid isolating it too much. Anyway, the core functionality here is the Tool Call search_catalog_local which uses the RAG to call for matches in the vector database for the specific site_id to provide around 3 product suggestions. Also, the suggested product names are pass along the graph to the main logic that runs the POST call to match these products in the original (site-specific) JSON file that can provide all the details back through the API response. This avoids passing lots of data back to the LLM where token usage can blow up.

Finally, the step 4 builds comprohensive answer combining:
- The LLM polite and thorough answer which product is mostly recommended and why.
- The data structures of the products retrieved. This carries a ton of value for any application consuming the API as the structured data can be the base for many recommendation features and following steps to acheive the purchase.

![Getting Started](Diagram.png)

### API Endpoint:

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

```
{
    "site_id": 
    "query":
    "conversation_id": optional
}
```

It accessed the graph built with LangGraph to be able to activate the agent that can process what is asked as 
Human message and use its tool (mainly RAG tool) to inspect if a product/s can match the request.
This process expects the agent to analyze the retrieve results from the dataset and produce an polite and useful
answer as it will be the expected string to be displayed in the chat for the end user.

The response has the following JSON structure:

```
{
    "answer": "string",
    "retrieved_products": []
    "conversation_id": optional
}
```

## Setup and Execution
The current solution can be easily deployment and tested with a Uvicorn server running locally and a couple of API keys for remote features (including of course the model).



## Decisions and Trade-offs
Explain the key decisions you made in designing your agent and retrieval pipeline. What trade-offs did you consciously accept for the sake of this PoC, and what would you change or recommend for a production deployment?

The first big decision was the selection of the React agent template from LangGraph GitHub repos to enable the main chatbot features. This decision marries the PoC to the structure that LangGraph proposes but also brings lots of off-the-shelf benefits, like the distribution of responsibilities in the different python scripts in the project, ease of connection to other features in and out of the project, a json configuration file "langgrash.json" that can later be adapted to use LangSmith Studio for better observability of the agents performance, and more.
Also, a big pro in the list for LangGraph is the scalability of the current graph into more complex scenarios where the chatbot can produce sub-agents to complete other major procedure like doing a complete checkout for a user or just scaling the tools like a Agentic RAG instead of the current one.

The second big decision was to split the dataset into site specific datasets to avoid mixing product from other sites in the conversation. This basically from the assumption that a user will be accessing a single site in this search for the desired product and taking that user to another site is not within the possibilities for this chatbot. If this changing to a multi-site chatbot that can cross-reference products between them, that may cost a larger refactor to the way this tool is implemented. So it was a bit of an effort when unplugging the LLM from the large number of details in the dataset (as described in the diagram above), to avoid mixing products from one site into the other in the varios checks and searches that happen in the process. The downside is how it becomes less flexible to change sites within the same conversation or allowing to recommend products of the same original dataset but different sites from the current one.

The third decision was to incorporate a conversation_id value generated by the application to enable a coherent conversation that can be decided as well by the Client side. The conversation thread can be kept consistent by the client repeating the id provided or providing another id provided before (even though that resets if the application resets because the conversation history is not saved externally yet). LangGraph provided options to enable memory between calls with "configurable threah Ids" that can keep track of the conversation (including tool calls) and that can be easily visible from LangSmith. I decided for a more straightforward solution that still kept the JSON structures proposed the in the assignment specs definition.

## Future Roadmap
If this PoC were approved, what would be the next 3-5 steps to move it towards production? (Think about testing, observability, evaluation, scalability, additional features, etc.)

- Observability: as mentioned above this project is not hard to connect to LangSmith where a whole suit of observability and testing tools are available. 

- Testing: the different steps (especially the ones dealing with datasets) should include testing in different levels to check that things are runnig as expected. Also, depending on which integrations are added to this PoC, the corresponding tests should be included. The "\health" route implemented in the API is part of the effort to be able to test along the way of implementation.

- Scalability:

- Additional features:

