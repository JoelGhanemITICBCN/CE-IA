from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

pc = Pinecone(api_key="")


# Define a sample dataset where each item has a unique ID, text, and category
data = [
    {
        "id": "rec1",
        "text": "Apples are a great source of dietary fiber, which supports digestion and helps maintain a healthy gut.",
        "category": "digestive system" 
    },
    {
        "id": "rec2",
        "text": "Apples originated in Central Asia and have been cultivated for thousands of years, with over 7,500 varieties available today.",
        "category": "cultivation"
    },
    {
        "id": "rec3",
        "text": "Rich in vitamin C and other antioxidants, apples contribute to immune health and may reduce the risk of chronic diseases.",
        "category": "immune system"
    },
    {
        "id": "rec4",
        "text": "The high fiber content in apples can also help regulate blood sugar levels, making them a favorable snack for people with diabetes.",
        "category": "endocrine system"
    }
]

# Convert the text into numerical vectors that Pinecone can index
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d["text"] for d in data],
    parameters={
        "input_type": "passage", 
        "truncate": "END"
    }
)

# Create a serverless index
index_name = "example-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    ) 

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Target the index
# In production, target an index by its unique DNS host, not by its name
# See https://docs.pinecone.io/guides/data/target-an-index
index = pc.Index(index_name)

# Prepare the records for upsert
# Each contains an 'id', the vector 'values', 
# and the original text and category as 'metadata'
records = []
for d, e in zip(data, embeddings):
    records.append({
        "id": d["id"],
        "values": e["values"],
        "metadata": {
            "source_text": d["text"],
            "category": d["category"]
        }
    })

# Upsert the records into the index
index.upsert(
    vectors=records,
    namespace="example-namespace"
)

# Define your query
query = "Health risks"

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[query],
    parameters={
        "input_type": "query"
    }
)

# Search the index for the three most similar vectors
results = index.query(
    namespace="example-namespace",
    vector=query_embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

# Rerank the search results based on their relevance to the query
ranked_results = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query="Health risks",
    documents=[
        {"id": "rec3", "source_text": "Rich in vitamin C and other antioxidants, apples contribute to immune health and may reduce the risk of chronic diseases."},
        {"id": "rec1", "source_text": "Apples are a great source of dietary fiber, which supports digestion and helps maintain a healthy gut."},
        {"id": "rec4", "source_text": "The high fiber content in apples can also help regulate blood sugar levels, making them a favorable snack for people with diabetes."}
    ],
    top_n=3,
    rank_fields=["source_text"],
    return_documents=True,
    parameters={
        "truncate": "END"
    }
)

# Search the index with a metadata filter
filtered_results = index.query(
    namespace="example-namespace",
    vector=query_embedding.data[0].values,
    filter={
        "category": {"$eq": "digestive system"}
        },
    top_k=3,
    include_values=False,
    include_metadata=True
)


print(filtered_results)
