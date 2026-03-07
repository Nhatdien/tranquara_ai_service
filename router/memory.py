"""
AI Memory REST Endpoints

These endpoints are called by the Go backend (internal) to manage
memory vectors in Qdrant alongside PostgreSQL records.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from database.vector_database import index_memory, delete_memory

router = APIRouter()


class IndexMemoryRequest(BaseModel):
    """Request to index a single memory into Qdrant."""
    memory_id: str
    user_id: str
    content: str
    category: str = "preferences"
    confidence: float = 0.5
    created_at: Optional[str] = None


class DeleteMemoryRequest(BaseModel):
    """Request to delete a memory from Qdrant."""
    memory_id: str


@router.post("/api/internal/memory/index")
async def index_memory_endpoint(request: IndexMemoryRequest):
    """
    Index a memory into Qdrant for RAG retrieval.
    Called by Go backend after storing in PostgreSQL.
    """
    try:
        index_memory(
            memory_id=request.memory_id,
            user_id=request.user_id,
            content=request.content,
            category=request.category,
            confidence=request.confidence,
            created_at=request.created_at,
        )
        return {"status": "indexed", "memory_id": request.memory_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Qdrant indexing error: {str(e)}")


@router.delete("/api/internal/memory/{memory_id}")
async def delete_memory_endpoint(memory_id: str):
    """
    Delete a memory from Qdrant.
    Called by Go backend when user deletes a memory.
    """
    try:
        delete_memory(memory_id=memory_id)
        return {"status": "deleted", "memory_id": memory_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Qdrant deletion error: {str(e)}")
