from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
import os

# Connect to your running Qdrant instance
qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
client = QdrantClient(host=qdrant_host, port=qdrant_port)

# --- Journal Entries Collection (for RAG in Go Deeper) ---
JOURNAL_COLLECTION = "journal_entries"
VECTOR_SIZE = 1536  # OpenAI text-embedding-ada-002
DISTANCE_METRIC = Distance.COSINE

if not client.collection_exists(JOURNAL_COLLECTION):
    print(f"Creating collection: {JOURNAL_COLLECTION}")
    client.recreate_collection(
        collection_name=JOURNAL_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE_METRIC)
    )
else:
    print(f"Collection '{JOURNAL_COLLECTION}' already exists.")

embeddings = OpenAIEmbeddings()

journal_vector_store = QdrantVectorStore(
    embedding=embeddings,
    collection_name=JOURNAL_COLLECTION,
    client=client
)


def search_user_journals(user_id: str, query: str, top_k: int = 5) -> list:
    """
    Search for past journals by a specific user that are semantically similar to the query.

    Args:
        user_id: The user's UUID string
        query: The text to search for similar journals
        top_k: Number of results to return

    Returns:
        List of matching Documents with metadata
    """
    results = journal_vector_store.similarity_search(
        query=query,
        k=top_k,
        filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
    )
    return results


def index_journal(journal_id: str, user_id: str, content: str,
                  title: str = "", mood_score: int = None,
                  mood_label: str = None, created_at: str = None):
    """
    Index a journal entry into Qdrant for future RAG retrieval.
    Uses journal_id as the point ID to enable upsert (re-indexing on update).

    Args:
        journal_id: UUID of the journal (used as Qdrant point ID for upsert)
        user_id: UUID of the user who owns this journal
        content: The journal text content to embed
        title: Journal title
        mood_score: 1-10 mood rating
        mood_label: Mood label (e.g. "Sunny", "Storm")
        created_at: ISO timestamp of journal creation
    """
    from langchain_core.documents import Document

    # Build the text to embed: title + content for richer semantic meaning
    embed_text = f"{title}\n{content}" if title else content

    doc = Document(
        page_content=embed_text,
        metadata={
            "journal_id": journal_id,
            "user_id": user_id,
            "title": title,
            "mood_score": mood_score,
            "mood_label": mood_label,
            "created_at": created_at,
            "type": "journal"
        }
    )

    # Use journal_id as the vector point ID so updates overwrite the old entry
    journal_vector_store.add_documents(
        documents=[doc],
        ids=[journal_id]
    )
    print(f"Indexed journal {journal_id} for user {user_id}")


def delete_journal(journal_id: str):
    """
    Delete a journal entry vector from Qdrant.
    Called when a user deletes a journal so it no longer appears in RAG results.

    Args:
        journal_id: UUID of the journal (the Qdrant point ID)
    """
    from qdrant_client.models import PointIdsList

    client.delete(
        collection_name=JOURNAL_COLLECTION,
        points_selector=PointIdsList(points=[journal_id])
    )
    print(f"Deleted journal {journal_id} from Qdrant")
