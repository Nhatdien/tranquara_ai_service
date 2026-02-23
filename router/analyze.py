from fastapi import APIRouter, HTTPException
from service.ai_service_processor import AIProcessor
from models.messages import AnalyzeJournalRequest

router = APIRouter()


class AnalyzeJournalResponse:
    """Simple response wrapper — FastAPI will serialize it from the dict."""
    pass


@router.post("/api/analyze-journal")
async def analyze_journal(request: AnalyzeJournalRequest):
    """
    Generate a single follow-up question based on user's journal content.
    Now enhanced with RAG: queries Qdrant for the user's past journals
    and includes them as context for more personalized, pattern-aware questions.
    """
    try:
        ai_processor = AIProcessor()

        question = ai_processor.generate_journal_question(
            user_id=request.user_id,
            content=request.content,
            mood_score=request.mood_score,
            slide_prompt=request.slide_prompt,
            slide_group_context=request.slide_group_context,
            current_slide_id=request.current_slide_id,
            collection_title=request.collection_title,
            direction=request.direction
        )

        return {"question": question}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI service error: {str(e)}")
