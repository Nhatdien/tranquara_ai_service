from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from models.user import UserInformations

T = TypeVar("T")


class SyncDataMessage(BaseModel, Generic[T]):
    event: str
    timestamp: str = Field(default=datetime.now().isoformat())
    payload: T


class SyncJournalPayload(BaseModel):
    user_id: str
    title: str
    content: str


class SyncChatlogPayload(BaseModel):
    user_id: str
    sender_type: str
    journal_id: str
    message: str


class SyncEmotionLog(BaseModel):
    user_id: str
    emotion: str
    source: str
    context: str


# --- AI Tasks (Core → AI service via ai_tasks queue) ---

class AITaskMessage(BaseModel):
    """Envelope for messages received from the ai_tasks queue."""
    event: str
    timestamp: str
    payload: Any


class JournalIndexPayload(BaseModel):
    """Payload for journal.index events — sent by core service for Qdrant indexing."""
    id: str
    user_id: str
    title: str
    content: str
    mood_score: Optional[int] = None
    mood_label: Optional[str] = None
    created_at: Optional[str] = None


class JournalDeletePayload(BaseModel):
    """Payload for journal.delete events — removes journal vector from Qdrant."""
    id: str
    user_id: str


# --- Analyze Journal (Go Deeper) ---

class AnalyzeJournalRequest(BaseModel):
    """Request model for the /api/analyze-journal endpoint."""
    user_id: str
    content: str
    mood_score: int
    slide_prompt: Optional[str] = None
    slide_group_context: Optional[dict] = None
    current_slide_id: Optional[str] = None
    collection_title: Optional[str] = None
    direction: Optional[str] = None
