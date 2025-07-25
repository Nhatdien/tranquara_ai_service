from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

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
    message: str


class SyncEmotionLog(BaseModel):
    user_id: str
    emotion: str
    source: str
    context: str
