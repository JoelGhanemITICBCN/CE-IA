from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,AutoTokenizer,pipeline
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from haystack import Document
from haystack.components.readers import ExtractiveReader
import time
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

def chunk_text(text, max_tokens=364):
    tokens = tokenizer.encode(text,max_length=512,truncation=True,add_special_tokens=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0,len(tokens),max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]
pc = Pinecone(api_key="pcsk_4v9hx7_DhvDqNNZY8zCmqwrs1BmfYz4noUTRoveoqoenXrrDtwoG7WQqFBijwXQYUkeWKH")


dataset = load_dataset("dmntrd/QuijoteFullText",split="train")
chunked_texts =[chunk for doc in dataset for chunk in chunk_text(doc["text"])]
#documents = [Document(content=doc["content"],meta=doc["meta"]) for doc in dataset]

# Convert the text into numerical vectors that Pinecone can index
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(model_name)
embeddings = model.encode(chunked_texts,convert_to_numpy=True)

# Create a serverless index
index_name = "example-index"

existing_indexes = pc.list_indexes()

pc.delete_index(index_name)
time.sleep(10)

pc.create_index(
    name=index_name,
    dimension=768,
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
print(pc.list_indexes())
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
            "source_text": d,  # Guarda el fragmento original
            "category": "Quijote"
        }
    })


# Upsert the records into the index
batch_size = 100
for i in range(0,len(records),batch_size):
    batch = records[i:i + batch_size]
    index.upsert(vectors=batch,namespace="example-namespace")

# Define your query
query = "Describe a Don Quijote"

# Convert the query into a numerical vector that Pinecone can search with
query_embedding = model.encode([query], convert_to_numpy=True)

index_stats = index.describe_index_stats()
print(index_stats)

if "example-namespace" in index_stats["namespaces"]:
    print(f"El namespace 'example-namespace' tiene {index_stats['namespaces']['example-namespace']['vector_count']} vectores.")
else:
    print("El namespace 'example-namespace' no existe. Intenta indexar de nuevo.")

# Search the index for the three most similar vectors
results = index.query(
    namespace = "example-namespace",
    vector=query_embedding[0].tolist(),
    top_k=25,
    include_values=False,
    include_metadata=True,
    timeout=50
)
print("Resultados de busqueda: ",results)

retrieved_docs = []
for match in results["matches"]:
    if "metadata" in match and "source_text" in match["metadata"]:
        retrieved_docs.append({
            "id":match["id"],
            "source_text" : match["metadata"]["source_text"]
            })
# Rerank the search results based on their relevance to the query
if retrieved_docs:
    ranked_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents= retrieved_docs,
        top_n=50,
        rank_fields=["source_text"],
        return_documents=True,
        parameters={
            "truncate": "END"
        }
    )
    print("Resultados rankeados: ",ranked_results)
else :
    print("No hay resultados")

# Search the index with a metadata filter
filtered_results = index.query(
    namespace="example-namespace",
    vector=query_embedding[0].tolist(),
    filter={
        "category": "Quijote"
        },
    top_k=3,
    include_values=False,
    include_metadata=True
)


print(filtered_results)
"""


relevant_texts = [doc["document"]["source_text"] for doc in ranked_results.data]

gen_model_name="google/flan-t5-large"
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

context = "\n\n".join(relevant_texts)
#prompt = f0"

""
#Context:
    #{context}
#
#Pregunta: {query}
#Genera una respuesta basada en el contexto anterior.

input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids
output = gen_model.generate(input_ids, max_length=200)
response = gen_tokenizer.decode(output[0], skip_special_tokens=True)

print("\n Respuesta Generada:")
print(response)
"""
