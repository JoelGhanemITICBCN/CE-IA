from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from haystack import Document
from haystack.components.readers import ExtractiveReader
import time
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

def chunk_text(text, max_tokens=96):
    tokens = tokenizer.encode(text,max_length=512,truncation=True,add_special_tokens=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0,len(tokens),max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]
pc = Pinecone(api_key="")


dataset = load_dataset("dmntrd/QuijoteFullText",split="train")
print("chunked texts")
chunked_texts =[chunk for doc in dataset for chunk in chunk_text(doc["text"])]
print(chunked_texts)
#documents = [Document(content=doc["content"],meta=doc["meta"]) for doc in dataset]

# Convert the text into numerical vectors that Pinecone can index
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
embeddings = model.encode(chunked_texts,convert_to_numpy=True)

# Create a serverless index
index_name = "example-index"

if not pc.list_indexes():
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

# Prepare the records for upsert
records = []
for i, (d, e) in enumerate(zip(chunked_texts, embeddings)):  # Usa chunked_texts en lugar de dataset
    records.append({
        "id": str(i),  # Usa el índice como ID
        "values": e.tolist(),  # Convierte a lista si es un array de NumPy
        "metadata": {
            "source_text": d  # Guarda el fragmento original
        }
    })


# Upsert the records into the index
batch_size = 100
for i in range(0,len(records),batch_size):
    batch = records[i:i + batch_size]
    index.upsert(vectors=batch,namespace="example-namespace")

# Define your query
query = "which is the most interesting aspecto of the main character"

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = pc.inference.embed(
   model="text-embedding-ada-002",
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
