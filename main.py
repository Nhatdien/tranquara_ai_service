import asyncio
import json
import sys
import dotenv
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.rabbitmq import rabbitmq_conn
from models.messages import AITaskMessage, JournalIndexPayload, JournalDeletePayload
from database.vector_database import index_journal, delete_journal
from router.analyze import router as analyze_router


dotenv.load_dotenv()


# --- RabbitMQ Consumer: ai_tasks queue ---

def ai_tasks_callback(ch, method, properties, body):
    """
    Process messages from the ai_tasks queue.
    Currently handles:
      - journal.index: Index/re-index a journal entry in Qdrant
    """
    try:
        raw = json.loads(body)
        message = AITaskMessage.model_validate(raw)

        if message.event == "journal.index":
            payload = JournalIndexPayload.model_validate(message.payload)
            index_journal(
                journal_id=payload.id,
                user_id=payload.user_id,
                content=payload.content,
                title=payload.title,
                mood_score=payload.mood_score,
                mood_label=payload.mood_label,
                created_at=payload.created_at,
            )
            print(
                f"[ai_tasks] Indexed journal {payload.id} for user {payload.user_id}")

        elif message.event == "journal.delete":
            payload = JournalDeletePayload.model_validate(message.payload)
            delete_journal(journal_id=payload.id)
            print(
                f"[ai_tasks] Deleted journal {payload.id} for user {payload.user_id}")

        else:
            print(f"[ai_tasks] Unknown event: {message.event}")

    except Exception as e:
        print(f"[ai_tasks] Error processing message: {e}")


async def start_rabbitmq_consumer():
    """Start the blocking RabbitMQ consumer in a background thread."""
    rabbitmq_conn.channel.queue_declare("ai_tasks")

    loop = asyncio.get_event_loop()

    def run():
        try:
            print("[ai_tasks] Starting RabbitMQ consumer on 'ai_tasks' queue...")
            rabbitmq_conn.consume(queue_name='ai_tasks',
                                  callback=ai_tasks_callback)
        except Exception as e:
            print(f"[ai_tasks] RabbitMQ consumer error: {e}")
            sys.exit(1)

    await loop.run_in_executor(None, run)


# --- FastAPI App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch RabbitMQ consumer
    consumer_task = asyncio.create_task(start_rabbitmq_consumer())
    print("[startup] RabbitMQ consumer started")
    yield
    # Shutdown
    consumer_task.cancel()
    print("[shutdown] RabbitMQ consumer stopped")


app = FastAPI(lifespan=lifespan)

# Register routers
app.include_router(analyze_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "service": "tranquara_ai_service"}


if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("Server interrupted.")
        sys.exit(0)
