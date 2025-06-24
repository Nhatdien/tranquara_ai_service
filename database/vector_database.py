from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Connect to your running Qdrant instance
client = QdrantClient(host="qdrant", port=6333)

# Define your collection name and vector settings
collection_name = "my_collection"
vector_size = 1536
distance_metric = Distance.COSINE

# Check if collection exists
if not client.collection_exists(collection_name):
    print(f"Creating collection: {collection_name}")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_metric)
    )
else:
    print(f"Collection '{collection_name}' already exists.")


qdrant = QdrantVectorStore(
    embedding=OpenAIEmbeddings(),
    collection_name=collection_name,
    client=client
)


# Define crud vector funcitons in here
