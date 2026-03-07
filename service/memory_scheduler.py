"""
AI Memory Generation Scheduler

Periodically extracts factual insights from user journals using GPT.
Runs every 12 hours, processes users with new journal activity.

Flow:
  1. Fetch active users (with recent journal activity) from Go backend
  2. For each user: fetch recent journals + existing memories
  3. Send to GPT for extraction → deduplicate → store in PostgreSQL + Qdrant
"""

import os
import httpx
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from service.ai_service_processor import AIProcessor
from database.vector_database import index_memory, get_all_user_memories

# Go backend internal API base URL
CORE_SERVICE_URL = os.getenv("CORE_SERVICE_URL", "http://core-service:4000")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")

# Scheduler interval (hours)
MEMORY_GENERATION_INTERVAL_HOURS = int(
    os.getenv("MEMORY_INTERVAL_HOURS", "12"))

scheduler = AsyncIOScheduler()


async def _fetch_active_users(since: str) -> list[str]:
    """Fetch user IDs with recent journal activity from Go backend."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{CORE_SERVICE_URL}/v1/internal/active-journal-users",
                params={"since": since},
                headers={"X-Internal-Key": INTERNAL_API_KEY},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("user_ids", [])
    except Exception as e:
        print(f"[memory-scheduler] Error fetching active users: {e}")
        return []


async def _fetch_user_journals(user_id: str, since: str) -> list[dict]:
    """Fetch recent journals for a user from Go backend."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{CORE_SERVICE_URL}/v1/internal/user-journals",
                params={"user_id": user_id, "since": since},
                headers={"X-Internal-Key": INTERNAL_API_KEY},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("journals", [])
    except Exception as e:
        print(f"[memory-scheduler] Error fetching journals for {user_id}: {e}")
        return []


async def _fetch_existing_memories(user_id: str) -> list[dict]:
    """Fetch existing memories for a user from Go backend."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{CORE_SERVICE_URL}/v1/internal/ai-memories/{user_id}",
                headers={"X-Internal-Key": INTERNAL_API_KEY},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("memories", [])
    except Exception as e:
        print(f"[memory-scheduler] Error fetching memories for {user_id}: {e}")
        return []


async def _store_memories(user_id: str, memories: list[dict]) -> list[dict]:
    """Store new memories in Go backend (PostgreSQL) and return created records with IDs."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{CORE_SERVICE_URL}/v1/internal/ai-memories/batch",
                json={"user_id": user_id, "memories": memories},
                headers={
                    "X-Internal-Key": INTERNAL_API_KEY,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("created", [])
    except Exception as e:
        print(f"[memory-scheduler] Error storing memories for {user_id}: {e}")
        return []


async def process_user_memories(user_id: str, since: str):
    """
    Process a single user: fetch journals, extract memories, store + index.
    """
    try:
        # 1. Fetch recent journals
        journals = await _fetch_user_journals(user_id, since)
        if not journals:
            return

        # 2. Fetch existing memories (for dedup prompt context)
        existing = await _fetch_existing_memories(user_id)
        existing_contents = [m.get("content", "") for m in existing]

        # 3. Extract new memories via GPT
        ai_processor = AIProcessor()
        new_memories = ai_processor.extract_memories(
            user_id=user_id,
            journal_entries=journals,
            existing_memories=existing_contents,
        )

        if not new_memories:
            return

        # 4. Store in PostgreSQL via Go backend
        created = await _store_memories(user_id, new_memories)

        # 5. Index in Qdrant for RAG
        for memory in created:
            memory_id = memory.get("id")
            if memory_id:
                index_memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    content=memory.get("content", ""),
                    category=memory.get("category", "preferences"),
                    confidence=memory.get("confidence", 0.5),
                    created_at=memory.get("created_at"),
                )

        print(
            f"[memory-scheduler] User {user_id}: {len(created)} memories created + indexed")

    except Exception as e:
        print(f"[memory-scheduler] Error processing user {user_id}: {e}")


async def run_memory_generation():
    """
    Main periodic job: find active users and extract memories.
    Runs every 12 hours.
    """
    print(
        f"[memory-scheduler] Starting memory generation cycle at {datetime.now(timezone.utc).isoformat()}")

    since = (datetime.now(timezone.utc) -
             timedelta(hours=MEMORY_GENERATION_INTERVAL_HOURS)).isoformat()

    # 1. Get users with recent journal activity
    user_ids = await _fetch_active_users(since)
    if not user_ids:
        print("[memory-scheduler] No active users found, skipping cycle")
        return

    print(f"[memory-scheduler] Processing {len(user_ids)} active users")

    # 2. Process each user (sequentially to avoid rate limits)
    success_count = 0
    error_count = 0
    for user_id in user_ids:
        try:
            await process_user_memories(user_id, since)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"[memory-scheduler] Failed for user {user_id}: {e}")

    print(
        f"[memory-scheduler] Cycle complete: {success_count} success, {error_count} errors")


def start_scheduler():
    """Initialize and start the memory generation scheduler."""
    scheduler.add_job(
        run_memory_generation,
        trigger="interval",
        hours=MEMORY_GENERATION_INTERVAL_HOURS,
        id="memory_generation",
        name="AI Memory Generation",
        replace_existing=True,
        next_run_time=None,  # Don't run immediately on startup
    )
    scheduler.start()
    print(
        f"[memory-scheduler] Scheduler started (every {MEMORY_GENERATION_INTERVAL_HOURS}h)")


def stop_scheduler():
    """Gracefully stop the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[memory-scheduler] Scheduler stopped")
