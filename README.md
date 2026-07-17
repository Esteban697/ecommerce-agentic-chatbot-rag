## High-Level Design

The following diagram shows both how the different components connect to each other (without going inside of the LangGraph based graph that builds the agent) and also shows the expected sequence of components that run when using the Uvicorn server to run the PoC:

* First, the execution takes us to the data ingestion stage that takes the JSON file with the product catalog that the diagram represents as Dataset 0. This will be parsed and get rid of duplicated products (detected in our first analysis of the dataset). After algorithmic cleaning and following step will produce one new JSON file per each site_id provided inside Dataset 0. This allows for a cleaner data management without the concern of mixing products between sites. Next, each JSON file will be treated as the source for embeddings and vector database to produce. The Nomic model persforms a remote inference to produce the embeddings and these are passed on to Chroma to produce a dedicated vector database for that group of products of a specific site. This will be useful when retrieval occurs.

* Second, when the data is locally saved and prepared for consumption, the user can start sending queries through the API implemented with FastAPI. The POST call needs to provide a JSON with query and site_id (and optional conversation_id normally from second message onwards). If you provide the conversation_id that comes in the first response of this API call, you can enjoy a conversation the chatbot will follow with semantic memory.

* Third, when the Chatbot agent receives queries, the LLM (in this case "openai/gpt-4.1") will decide when and which tool to call based on the system prompt that describes in detail what are the guardrails set for this chatbot to focus on product recommendation. The web_search tool based on Tavily is provided as a means to keep the chatbot situationally aware (even though a current_time value is passed to the system prompt) to avoid isolating it too much. Anyway, the core functionality here is the Tool Call search_catalog_local which uses the RAG to call for matches in the vector database for the specific site_id to provide around 3 product suggestions. Also, the suggested product names are pass along the graph to the main logic that runs the POST call to match these products in the original (site-specific) JSON file that can provide all the details back through the API response. This avoids passing lots of data back to the LLM where token usage can blow up.

* Finally, the step 4 builds comprohensive answer combining:
    * The LLM polite and thorough answer which product is mostly recommended and why.
    * The data structures of the products retrieved. This carries a ton of value for any application consuming the API as the structured data can be the base for many recommendation features and following steps to acheive the purchase.

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
NOTE: You can get different answer using different LLMs and even using the same one.

Follow the steps in your Git Bash terminal to configure the virtual environment to run the server and try the PoC:
- Navigate to the project path
```
cd /path/to/your/project/ecommerce-agentic-chatbot-rag
```

- Include these keys in your `.env` with the usual command so the virtual environment pull them with `load_dotenv()` when running `main.py`.
```
echo "API_KEY=your_secret_key_here" >> .env
```
The first key is Open AI API Key (it expires after this PoC is discussed and has limited funds so I have no problem to share it)
```
OPENAI_API_KEY="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"
```
The second key is Nomic API Key
```
NOMIC_API_KEY="nk-BiKfeOTbLxdppQVzOFphz4P1XzSwpJjbsoAfhTegu0g"
```
The last key is Tavily API Key (free)
```
TAVILY_API_KEY="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"
```

- Create the virtual environment (named 'venv')
```
python -m venv venv
```

- Activate it and run the server

The following commands to activate the virtual environment and running the Uvicorn server:
```
cd Videos/ZooPlus/ecommerce-agentic-chatbot-rag/.venv/Script
```

```
source activate
```

```
python -m uvicorn react_agent.main:app --reload
```

You will see the loading and in your Git Bash terminal you will see the prints from the app showing what it happening in each step like this:

![Getting Started](Capture_terminal.png)

- Open up the URL provided and add the `/docs` after it to have the API details and easy way to test it 
Copy the following test query and send it pressing *Execute*
```
{
    "site_id": 1,
    "query": "Do you have dry food for a puppy with a sensitive stomach in your store?"
}
```
![Getting Started](Capture_browser.png)


## Decisions and Trade-offs
Explain the key decisions you made in designing your agent and retrieval pipeline. What trade-offs did you consciously accept for the sake of this PoC, and what would you change or recommend for a production deployment?

The first big decision was the selection of the **React agent template from LangGraph** GitHub repos to enable the main chatbot features. This decision marries the PoC to the structure that LangGraph proposes but also brings lots of off-the-shelf benefits, like the distribution of responsibilities in the different python scripts in the project, ease of connection to other features in and out of the project, a json configuration file "langgrash.json" that can later be adapted to use LangSmith Studio for better observability of the agents performance, and more.
Also, a big pro in the list for LangGraph is the scalability of the current graph into more complex scenarios where the chatbot can produce sub-agents to complete other major procedure like doing a complete checkout for a user or just scaling the tools like a Agentic RAG instead of the current one.

The second big decision was to **split the dataset into site specific datasets** to avoid mixing product from other sites in the conversation. This basically from the assumption that a user will be accessing a single site in this search for the desired product and taking that user to another site is not within the possibilities for this chatbot. If this changing to a multi-site chatbot that can cross-reference products between them, that may cost a larger refactor to the way this tool is implemented. So it was a bit of an effort when unplugging the LLM from the large number of details in the dataset (as described in the diagram above), to avoid mixing products from one site into the other in the varios checks and searches that happen in the process. The downside is how it becomes less flexible to change sites within the same conversation or allowing to recommend products of the same original dataset but different sites from the current one.

The third decision was to **incorporate a conversation_id** value in the API request and responses generated by the application to enable a coherent conversation that can be decided as well by the Client side. The conversation thread can be kept consistent by the client repeating the id provided or providing another id provided before (even though that resets if the application resets because the conversation history is not saved externally yet). LangGraph provided options to enable memory between calls with "configurable threah Ids" that can keep track of the conversation (including tool calls) and that can be easily visible from LangSmith. I decided for a more straightforward solution that still kept the JSON structures proposed the in the assignment specs definition. Scalability and consistent state handling may have trouble with this, so it serves the purpose of PoC but not recommended for production.

## Future Roadmap

**Observability:** as mentioned above this project is not hard to connect to LangSmith where a whole suit of observability and testing tools are available. 

**Testing:** the different steps (especially the ones dealing with datasets) should include testing in different levels to check that things are runnig as expected. Also, depending on which integrations are added to this PoC, the corresponding tests should be included. The "\health" route implemented in the API is part of the effort to be able to test along the way of implementation.

**Scalability:** 
There are **two big concerns** when it comes to scalability:
**State handling:** Too many state handling attibutions to the API will make it harder to scale up allowing for several tenants. Better to move conversation memory and retrieved ID history out of process memory into Redis or a database, so the API becomes horizontally scalable. 

These are the parts that can be remove from process memory first:
    * conversation messages, retrieved product IDs, per-conversation site context, response summaries, optional user/session metadata.
That way the HTTP worker itself “forgets” everything after responding, even though the system still supports multi-turn chats.

**Latency:**
The LLM requests take some good time to be produced, especially when using the RAG pipeline retrieval so considering adding rate limiting, request tracing, and metrics on retrieval time, LLM time, and full-request latency, since scaling problems usually surface first as tail-latency issues rather than average latency issues.

For your current architecture, we can prioritize:
    * Shorten the system prompt and tool instructions, because every request pays that token cost.
    * Limit retrieved context aggressively, for example smaller k, shorter chunks, or a hard token budget on inserted context.
    * Use a faster model for most requests and reserve a slower, stronger model only for difficult cases.
    * Cache repeated or semantically similar results, especially for common catalog questions.

**Other concerns:**
**Startup and ingestion:** Ingestion runs at app startup so the API readiness becomes coupled to data preparation and embedding setup. That is manageable in development, but in production it hurts deploy speed, complicates autoscaling, and makes every new pod or worker repeat expensive initialization work. A big improvement we be keeping ingestion and vectorstore creation out of API startup, making them background jobs or deployment-time jobs.

**Additional features:** 
Storage is one big aspect that needs preparation before taking this PoC to production. **Changing the local storage solutions** picked for this PoC and making the cloud storage will be natural step to take for deployment of this solution. Remote embeddings like the one used here is one good example for what is needed, but the vector database can be an external resource to used for the RAG in production as well as the access to the JSON files. So that storage components replacement would be the first concern for moving all to the deployable and scalable infrastructure.

As mentioned above in the decisions, the **RAG pipeline can grow to become something more sophisticated.** A different datasets separation strategy can be implemented to include another global dataset that can be accessed as *site_0* or else. A different embedding can be chosen other than Nomic's. A different vector storage can be selected to change Chroma (I needed to update the library from LangGraph dealing with this type of database because of obsolete code). And the overall RAG pipeline architecture can evolve into something more dynamic: like Agentic RAG where the information retrieved is check by the same or other LLMs that can prepate the data before sending it back to the chatbot agent.

The graph that already works for conversational flows with product catalog searches **can start incorporating more complex tools**. LangGraph architecture allows for changing all modular parts and growing the graph with extra nodes and extra agents if necessary. It is prepared for it so it shouldn't take a large effort to add one by one more features to it over time.


