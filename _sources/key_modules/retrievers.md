# Retrievers

For more detailed usage information, please refer to our cookbook: [RAG Cookbook](../cookbooks/agents_with_rag.ipynb)

## 1. Concept
The Retrievers module is essentially a search engine. It's designed to help you find specific pieces of information by searching through large volumes of text. Imagine you have a huge library of books and you want to find where certain topics or keywords are mentioned, this module is like a librarian that helps you do just that.


## 2. Types

### 2.1. Vector Retriever
Vector Retriever typically refers to a method or system used in information retrieval and machine learning that utilizes vector representations of data. This approach is based on converting data, like text, images, or other forms of information, into numerical vectors in a high-dimensional space.

Here's a brief overview of how it works:

- Embedding Model: First, it uses an embedding model. This model turns text into a mathematical form (vectors).

- Storing Information: The module takes large documents, breaks them down into smaller chunks, and then uses the embedding model to turn these chunks into vectors. These vectors are stored in a vector storage.

- Retrieving Information: When we ask a question or make a query, the embedding model translates our question into vector and then searches in this vector storage for the closest matching vector. The closest matches are likely the most relevant pieces of information we are looking for.


### 2.2. Keyword Retriever
The Keyword Retriever is designed to retrieve relevant information based on keyword matching. Unlike the Vector Retriever that works with vector representations of data, the Keyword Retriever operates by identifying keywords or specific terms within documents and queries to find matches.

Here's a brief overview of how it works:

- Document Preprocessing: Before using the retriever, the documents are preprocessed to tokenize and index the keywords within them. Tokenization is the process of splitting text into individual words or phrases, making it easier to identify keywords.

- Query Parsing: When we input a question or query, the retriever parses the query to extract relevant keywords. This involves breaking down the query into its constituent terms.

- Keyword Matching: Once the keywords from the query are identified, the retriever searches the preprocessed documents for occurrences of these keywords. It checks for exact matches of the keywords within the documents.

- Ranking and Retrieval: After finding documents that contain the queried keywords, the retriever ranks these documents based on various factors, such as the frequency of keyword matches, document relevance, or other scoring methods. The top-ranked documents are then retrieved as the most relevant results.

## 3. Get Started

### 3.1. Using Vector Retriever

**Initialize VectorRetrieve:**
To get started, we need to initialize the `VectorRetriever` with an optional embedding model. If we don't provide an embedding model, it will use the default `OpenAIEmbedding`. Here's how to do it:
```python
from camel.embeddings import OpenAIEmbedding
from camel.retrievers import VectorRetriever

# Initialize the VectorRetriever with an embedding model
vr = VectorRetriever(embedding_model=OpenAIEmbedding())
```

**Embed and Store Data:**
Before we can retrieve information, we need to prepare the data and store it in vector storage. The `process` method takes care of this for us. It processes content from a file or URL, divides it into chunks, and stores their embeddings in the specified vector storage.
```python
# Provide the path to our content input (can be a file or URL)
content_input_path = "https://www.camel-ai.org/"

# Create or initialize a vector storage (e.g., QdrantStorage)
from camel.storages.vectordb_storages import QdrantStorage

vector_storage = QdrantStorage(
    vector_dim=OpenAIEmbedding().get_output_dim(),
    collection_name="my first collection",
    path="storage_customized_run",
)

# Embed and store chunks of data in the vector storage
vr.process(content_input_path, vector_storage)
```

**Execute a Query:**
Now that our data is stored, we can execute a query to retrieve information based on a search string. The `query` method executes the query and compiles the retrieved results into a string.
```python
# Specify our query string
query = "What is CAMEL"

# Execute the query and retrieve results
results = vr.query(query, vector_storage)
print(results)
```
```markdown
>>>  [{'similarity score': '0.812884257383057', 'content path': 'https://www.camel-ai.org/', 'metadata': {'filetype': 'text/html', 'languages': ['eng'], 'page_number': 1, 'url': 'https://www.camel-ai.org/', 'link_urls': ['/home', '/home', '/research/agent-trust', '/agent', '/data_explorer', '/chat', 'https://www.google.com/url?q=https%3A%2F%2Fcamel-ai.github.io%2Fcamel&sa=D&sntz=1&usg=AOvVaw1ifGIva9n-a-0KpTrIG8Cv', 'https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcamel-ai%2Fcamel&sa=D&sntz=1&usg=AOvVaw03Z2OD0-plx_zugZZgBb8w', '/team', '/sponsors', '/home', '/home', '/research/agent-trust', '/agent', '/data_explorer', '/chat', 'https://www.google.com/url?q=https%3A%2F%2Fcamel-ai.github.io%2Fcamel&sa=D&sntz=1&usg=AOvVaw1ifGIva9n-a-0KpTrIG8Cv', 'https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcamel-ai%2Fcamel&sa=D&sntz=1&usg=AOvVaw03Z2OD0-plx_zugZZgBb8w', '/team', '/sponsors', '/home', '/research/agent-trust', '/agent', '/data_explorer', '/chat', 'https://www.google.com/url?q=https%3A%2F%2Fcamel-ai.github.io%2Fcamel&sa=D&sntz=1&usg=AOvVaw1ifGIva9n-a-0KpTrIG8Cv', 'https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fcamel-ai%2Fcamel&sa=D&sntz=1&usg=AOvVaw03Z2OD0-plx_zugZZgBb8w', '/team', '/sponsors', 'https://github.com/camel-ai/camel'], 'link_texts': [None, 'Home', 'AgentTrust', 'Agent App', 'Data Explorer App', 'ChatBot', 'Docs', 'Github Repo', 'Team', 'Sponsors', None, 'Home', 'AgentTrust', 'Agent App', 'Data Explorer App', 'ChatBot', 'Docs', 'Github Repo', 'Team', 'Sponsors', 'Home', 'AgentTrust', 'Agent App', 'Data Explorer App', 'ChatBot', 'Docs', 'Github Repo', 'Team', 'Sponsors', None], 'emphasized_text_contents': ['Skip to main content', 'Skip to navigation', 'CAMEL-AI', 'CAMEL-AI', 'CAMEL:\xa0 Communicative Agents for "Mind" Exploration of Large Language Model Society', 'https://github.com/camel-ai/camel'], 'emphasized_text_tags': ['span', 'span', 'span', 'span', 'span', 'span']}, 'text': 'Search this site\n\nSkip to main content\n\nSkip to navigation\n\nCAMEL-AI\n\nHome\n\nResearchAgentTrust\n\nAgent App\n\nData Explorer App\n\nChatBot\n\nDocs\n\nGithub Repo\n\nTeam\n\nSponsors\n\nCAMEL-AI\n\nHome\n\nResearchAgentTrust\n\nAgent App\n\nData Explorer App\n\nChatBot\n\nDocs\n\nGithub Repo\n\nTeam\n\nSponsors\n\nMoreHomeResearchAgentTrustAgent AppData Explorer AppChatBotDocsGithub RepoTeamSponsors\n\nCAMEL:\xa0 Communicative Agents for "Mind" Exploration of Large Language Model Society\n\nhttps://github.com/camel-ai/camel'}]
```

### 3.2. Using Auto Retriever

To simplify the retrieval process further, we can use the `AutoRetriever` method. This method handles both embedding and storing data and executing queries. It's particularly useful when dealing with multiple content input paths.
```python
from camel.retrievers import AutoRetriever
from camel.types import StorageType

# Set the vector storage local path and the storage type
ar = AutoRetriever(vector_storage_local_path="camel/retrievers",storage_type=StorageType.QDRANT)

# Run the auto vector retriever
retrieved_info = ar.run_vector_retriever(
    contents=[
        "https://www.camel-ai.org/",  # Example remote url
    ],
    query="What is CAMEL-AI",
    return_detailed_info=True, # Set this as true is we want to get detailed info including metadata
)

print(retrieved_info)
```
```markdown
>>>  Original Query:
>>>  {What is CAMEL-AI}
>>>  Retrieved Context:
>>>  {'similarity score': '0.8380731206379989', 'content path': 'https://www.camel-ai.org/', 'metadata': {'emphasized_text_contents': ['Mission', 'CAMEL-AI.org', 'is an open-source community dedicated to the study of autonomous and communicative agents. We believe that studying these agents on a large scale offers valuable insights into their behaviors, capabilities, and potential risks. To facilitate research in this field, we provide, implement, and support various types of agents, tasks, prompts, models, datasets, and simulated environments.', 'Join us via', 'Slack', 'Discord', 'or'], 'emphasized_text_tags': ['span', 'span', 'span', 'span', 'span', 'span', 'span'], 'filetype': 'text/html', 'languages': ['eng'], 'link_texts': [None, None, None], 'link_urls': ['#h.3f4tphhd9pn8', 'https://join.slack.com/t/camel-ai/shared_invite/zt-1vy8u9lbo-ZQmhIAyWSEfSwLCl2r2eKA', 'https://discord.gg/CNcNpquyDc'], 'page_number': 1, 'url': 'https://www.camel-ai.org/'}, 'text': 'Mission\n\nCAMEL-AI.org is an open-source community dedicated to the study of autonomous and communicative agents. We believe that studying these agents on a large scale offers valuable insights into their behaviors, capabilities, and potential risks. To facilitate research in this field, we provide, implement, and support various types of agents, tasks, prompts, models, datasets, and simulated environments.\n\nJoin us via\n\nSlack\n\nDiscord\n\nor'}
```

That's it! We've successfully set up and used the Retriever Module to retrieve information based on queries from our data collection.

Feel free to customize the code according to your specific use case and data sources!
