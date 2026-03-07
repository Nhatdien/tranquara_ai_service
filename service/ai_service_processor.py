import os
import json
from dotenv import load_dotenv
from service.prompts import get_system_prompt
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from database.vector_database import (
    search_user_journals, search_user_memories,
    check_memory_duplicate, index_memory
)

load_dotenv()

# ─── Memory Extraction Prompt ──────────────────────────────────────────────

MEMORY_EXTRACTION_PROMPT = """You are analyzing journal entries to extract factual insights about the user.

Extract SHORT, FACTUAL statements about the user. Each statement should be:
- Written in first person (e.g., "I value...", "I enjoy...", "I struggle with...")
- One sentence maximum
- A genuine insight, NOT a summary of what they wrote
- Categorized as one of: values, habits, relationships, goals, struggles, preferences, patterns, growth

EXISTING MEMORIES (do NOT duplicate these):
{existing_memories}

JOURNAL ENTRIES TO ANALYZE:
{journal_entries}

Return a JSON array of new insights only:
[
  {{"content": "I value my family.", "category": "values", "confidence": 0.9}},
  {{"content": "Sleep quality drops when stressed about deadlines.", "category": "patterns", "confidence": 0.75}}
]

Rules:
- Only extract genuinely new insights not already covered by existing memories
- Confidence should reflect how clearly the journal supports this insight (0.5-1.0)
- Prefer fewer high-quality insights over many shallow ones
- Maximum 5 new insights per batch
- If no new insights can be extracted, return an empty array []
- Return ONLY valid JSON, no markdown formatting or code blocks"""


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

    def _retrieve_user_memories(self, user_id: str, current_content: str, top_k: int = 10) -> str:
        """
        Query Qdrant for the user's AI-generated memories that are relevant
        to the current journal content.

        Returns a formatted string of memory insights for prompt injection.
        """
        try:
            results = search_user_memories(
                user_id=user_id,
                query=current_content,
                top_k=top_k
            )

            if not results:
                return ""

            memory_lines = []
            for doc in results:
                category = doc.metadata.get("category", "general")
                memory_lines.append(f"  - [{category}] {doc.page_content}")

            return "\n".join(memory_lines)

        except Exception as e:
            print(f"[RAG] Error retrieving user memories: {e}")
            return ""

    def extract_memories(self, user_id: str, journal_entries: list[dict],
                         existing_memories: list[str]) -> list[dict]:
        """
        Extract new factual insights from journal entries using GPT.
        Performs semantic deduplication against existing memories.

        Args:
            user_id: User's UUID
            journal_entries: List of dicts with 'title', 'content', 'created_at'
            existing_memories: List of existing memory content strings (for prompt context)

        Returns:
            List of new unique memories: [{"content": "...", "category": "...", "confidence": 0.x}, ...]
        """
        if not journal_entries:
            return []

        # Format journal entries for the prompt
        formatted_journals = []
        for i, entry in enumerate(journal_entries, 1):
            title = entry.get("title", "Untitled")
            content = entry.get("content", "")[:1000]  # Truncate long entries
            date = entry.get("created_at", "Unknown date")
            formatted_journals.append(f"{i}. [{date}] \"{title}\"\n{content}")
        journals_text = "\n\n".join(formatted_journals)

        # Format existing memories for dedup context
        existing_text = "\n".join(f"- {m}" for m in existing_memories) if existing_memories else "(none yet)"

        # Build prompt
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            existing_memories=existing_text,
            journal_entries=journals_text
        )

        try:
            response = self.model.invoke([
                SystemMessage(content="You are a precise data extraction assistant. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])

            raw = response.content.strip()
            # Strip markdown code block if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            candidates = json.loads(raw)

            if not isinstance(candidates, list):
                print(f"[memories] GPT returned non-list: {type(candidates)}")
                return []

            # Validate and deduplicate
            valid_categories = {"values", "habits", "relationships", "goals",
                                "struggles", "preferences", "patterns", "growth"}
            new_memories = []

            for candidate in candidates[:5]:  # Max 5 per batch
                content = candidate.get("content", "").strip()
                category = candidate.get("category", "preferences")
                confidence = candidate.get("confidence", 0.5)

                if not content or len(content) < 5:
                    continue
                if category not in valid_categories:
                    category = "preferences"
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                confidence = max(0.0, min(1.0, float(confidence)))

                # Semantic dedup against Qdrant vectors
                if check_memory_duplicate(user_id, content):
                    continue

                new_memories.append({
                    "content": content,
                    "category": category,
                    "confidence": confidence,
                })

            print(f"[memories] User {user_id}: extracted {len(new_memories)} new memories "
                  f"from {len(journal_entries)} journals ({len(candidates) - len(new_memories)} duplicates skipped)")
            return new_memories

        except json.JSONDecodeError as e:
            print(f"[memories] JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"[memories] Error extracting memories: {e}")
            return []

    def generate_journal_question(self, user_id: str, content: str, mood_score: int,
                                  slide_prompt: str = None, slide_group_context: dict = None,
                                  current_slide_id: str = None, collection_title: str = None,
                                  direction: str = None, your_story: str = None) -> str:
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

        # --- RAG: Retrieve user memories ---
        user_memories_context = self._retrieve_user_memories(user_id, content)

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

        # --- Build user story section ---
        your_story_section = ""
        if your_story and your_story.strip():
            your_story_section = f"""
--- User's Personal Context ---
The user has shared the following about themselves. Use this to personalize your
question — acknowledge their situation, goals, or background when relevant.
Do NOT repeat their story back verbatim; weave it in naturally.

"{your_story.strip()}"
--- End Personal Context ---
"""

        # --- Build user memories section ---
        memories_section = ""
        if user_memories_context:
            memories_section = f"""
--- AI Memories (insights learned about this user) ---
These are factual insights extracted from the user's past journals.
Use them to ask more personalized, relevant questions. Reference naturally.

{user_memories_context}
--- End Memories ---
"""

        # --- Build the user prompt ---
        user_prompt = f"""Journaling Session Context:
{context_section}

Current Slide Prompt: {slide_prompt or "Free journaling"}

User's Current Writing:
{content}

User's Mood Score: {mood_score}/10
{your_story_section}{memories_section}{past_journals_section}
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
