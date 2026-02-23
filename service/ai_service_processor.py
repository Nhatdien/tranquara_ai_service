import os
from dotenv import load_dotenv
from service.prompts import get_system_prompt
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from database.vector_database import search_user_journals

load_dotenv()


class AIProcessor():
    """
    AI processor focused on generating RAG-enhanced journal follow-up questions.
    Uses Qdrant to retrieve user's past journals for richer, personalized guidance.
    """

    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False
        )

    def _retrieve_past_journals(self, user_id: str, current_content: str, top_k: int = 5) -> str:
        """
        Query Qdrant for the user's past journal entries that are semantically
        similar to what they're currently writing about.

        Returns a formatted string of past journal excerpts for prompt injection.
        """
        try:
            results = search_user_journals(
                user_id=user_id,
                query=current_content,
                top_k=top_k
            )

            if not results:
                return ""

            past_entries = []
            for i, doc in enumerate(results, 1):
                title = doc.metadata.get("title", "Untitled")
                mood = doc.metadata.get("mood_label") or doc.metadata.get(
                    "mood_score") or "N/A"
                date = doc.metadata.get("created_at", "Unknown date")
                # Truncate long content to keep prompt manageable
                excerpt = doc.page_content[:500]
                if len(doc.page_content) > 500:
                    excerpt += "..."

                past_entries.append(
                    f"  {i}. [{date}] \"{title}\" (mood: {mood})\n     {excerpt}"
                )

            return "\n".join(past_entries)

        except Exception as e:
            print(f"[RAG] Error retrieving past journals: {e}")
            return ""

    def generate_journal_question(self, user_id: str, content: str, mood_score: int,
                                  slide_prompt: str = None, slide_group_context: dict = None,
                                  current_slide_id: str = None, collection_title: str = None,
                                  direction: str = None) -> str:
        """
        Generate a single follow-up question based on journal content.
        Enhanced with:
        - Full slide group context for session awareness
        - RAG retrieval of past journals for personalized, pattern-aware questions

        Args:
            user_id: User's UUID for Qdrant filtering
            content: User's current journal text
            mood_score: User's mood rating (1-10)
            slide_prompt: Current slide question/prompt
            slide_group_context: Full slide group data including all slides
            current_slide_id: ID of the current slide being worked on
            collection_title: Name of the collection (e.g., "Daily Reflection")
            direction: Reflection direction ('why', 'emotions', 'patterns', 'challenge', 'growth')
        """
        # Get system prompt (with optional direction enhancement)
        system_prompt = get_system_prompt(direction)

        # --- RAG: Retrieve relevant past journals ---
        past_journals_context = self._retrieve_past_journals(user_id, content)

        # --- Build slide group context ---
        context_info = []

        if collection_title:
            context_info.append(f"Collection: {collection_title}")

        if slide_group_context:
            slide_group_title = slide_group_context.get(
                'title', 'Unknown Session')
            slide_group_desc = slide_group_context.get('description', '')
            context_info.append(f"Slide Group: {slide_group_title}")
            if slide_group_desc:
                context_info.append(f"Session Purpose: {slide_group_desc}")

            slides = slide_group_context.get('slides', [])
            if slides and len(slides) > 1:
                slide_questions = []
                for idx, slide in enumerate(slides, 1):
                    slide_type = slide.get('type', 'unknown')
                    question = slide.get('question', slide.get('title', ''))
                    is_current = (current_slide_id and slide.get(
                        'id') == current_slide_id)
                    marker = " [CURRENT SLIDE]" if is_current else ""
                    if question:
                        slide_questions.append(
                            f"  {idx}. [{slide_type}] {question}{marker}")
                if slide_questions:
                    context_info.append(
                        f"Full Session Flow:\n" + "\n".join(slide_questions))

        context_section = "\n".join(
            context_info) if context_info else "Free journaling session"

        # --- Build past journals section ---
        past_journals_section = ""
        if past_journals_context:
            past_journals_section = f"""
--- Past Journal Entries (semantically related) ---
The user has written about similar topics before. Use these to identify patterns,
recurring themes, emotional trends, or growth. Reference them naturally if relevant.

{past_journals_context}
--- End Past Journals ---
"""

        # --- Build the user prompt ---
        user_prompt = f"""Journaling Session Context:
{context_section}

Current Slide Prompt: {slide_prompt or "Free journaling"}

User's Current Writing:
{content}

User's Mood Score: {mood_score}/10
{past_journals_section}
Based on the FULL CONTEXT of this journaling session, the user's current writing,
and their past journal history (if available), generate ONE follow-up question that:
1. Relates specifically to what they just wrote
2. Stays aligned with the theme of this slide and the overall session
3. Helps them explore their thoughts and feelings more deeply
4. Feels natural and conversational
5. If past journals reveal patterns or recurring themes, gently reference them
   (e.g., "You mentioned something similar about work last week — what's changed?")

Generate the question now:"""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        question = response.content.strip()

        # Remove quotes if LLM added them
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1]

        return question
