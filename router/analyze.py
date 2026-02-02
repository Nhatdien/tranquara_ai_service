from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from service.ai_service_processor import AIProcessor

router = APIRouter()


class AnalyzeJournalRequest(BaseModel):
    content: str
    mood_score: int
    slide_prompt: Optional[str] = None
    slide_group_context: Optional[Dict[str, Any]
                                  ] = None  # Full slide group info
    current_slide_id: Optional[str] = None  # Current slide being worked on
    collection_title: Optional[str] = None  # Collection name for context
    # Reflection direction: 'why', 'emotions', 'patterns', 'challenge', 'growth'
    direction: Optional[str] = None


class AnalyzeJournalResponse(BaseModel):
    question: str


@router.post("/api/analyze-journal", response_model=AnalyzeJournalResponse)
async def analyze_journal(request: AnalyzeJournalRequest):
    """
    Generate a single follow-up question based on user's journal content and current slide prompt.
    Now includes full slide group context to make AI more aware of the journaling session.
    """
    try:
        # Create AIProcessor instance (without init_metadata for simple question generation)
        ai_processor = AIProcessor()

        # Generate question using the processor with enhanced context
        question = ai_processor.generate_journal_question(
            content=request.content,
            mood_score=request.mood_score,
            slide_prompt=request.slide_prompt,
            slide_group_context=request.slide_group_context,
            current_slide_id=request.current_slide_id,
            collection_title=request.collection_title,
            direction=request.direction
        )

        return AnalyzeJournalResponse(question=question)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI service error: {str(e)}")
