import asyncio
import json
import os
import sys
import dotenv
import uvicorn
import pika
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from service.rabbitmq import RabbitMQ
from service.ai_service_processor import AIProcessor
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from service.rabbitmq import rabbitmq_conn
from models.messages import SyncDataMessage, SyncChatlogPayload, InitConnectData, UserMessagePayload
from database.vector_database import QdrantClient
from datetime import datetime
from uuid import uuid4


dotenv.load_dotenv()
# --- RabbitMQ Consumer Task ---


async def start_rabbitmq_consumer():
    rabbitmq_conn.channel.queue_declare("ai_tasks")
    rabbitmq_conn.channel.queue_declare("sync_data")

    loop = asyncio.get_event_loop()

    def run():
        try:
            print("Starting RabbitMQ consumer...")
            rabbitmq_conn.consume(queue_name='ai_tasks',
                                  callback=rabbitmq_callback)
        except Exception as e:
            print("RabbitMQ error:", e)
            sys.exit(1)

    # Run blocking consumer in thread
    await loop.run_in_executor(None, run)


# --- FastAPI Startup Hook ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    consumer_task = asyncio.create_task(start_rabbitmq_consumer())
    print("RabbitMQ consumer started")

    yield

    # Shutdown logic (optional)
    consumer_task.cancel()
    print("Shutting down RabbitMQ consumer")

app = FastAPI(lifespan=lifespan)

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
connected_websockets: dict[str, WebSocket] = {}

# --- WebSocket Route ---


@app.websocket("/ws/{user_uuid}")
async def websocket_endpoint(websocket: WebSocket, user_uuid: str):
    await websocket.accept()
    connected_websockets[user_uuid] = websocket
    rabbitmq_ins = RabbitMQ()

    initial_message = await websocket.receive_text()
    init_metadata = InitConnectData.model_validate_json(initial_message)
    ai_processor = AIProcessor(init_metadata)
    try:
        while True:
            # Can be replaced with actual chat triggers

            data = await websocket.receive_text()
            # Decode the user message
            user_input = UserMessagePayload.model_validate_json(data)

            # Response with the model
            response = ai_processor.response_chat(user_input)
            response_message = {
                "content": response.content,
            }
            await connected_websockets[user_uuid].send_text(json.dumps(response_message))

            # sync data with Golang service
            for chat_message in [[data, "user"], [response.content, "bot"]]:
                chatlog = SyncChatlogPayload(
                    user_id=user_uuid, sender_type=chat_message[1], message=chat_message[0], journal_id=user_input.journal_id)
                message = SyncDataMessage[SyncChatlogPayload](
                    event="chatlog.create", payload=chatlog).model_dump_json()
                rabbitmq_ins.publish("sync_data", message)

            # Save to vector store
            documents_add = [Document(page_content=data, metadata={
                "type": "chatlog",
                "created_at": datetime.now().timestamp(),
                "sender_type": "user"
            }), Document(page_content=response.content, metadata={
                "type": "chatlog",
                "created_at": datetime.now().timestamp(),
                "sender_type": "bot"
            })]
            ai_processor.vector_store.add_documents(
                documents=documents_add, ids=[str(uuid4()) for _ in documents_add])

    except WebSocketDisconnect:
        del connected_websockets[user_uuid]
        print("WebSocket disconnected")


def rabbitmq_callback(ch, method, properties, body):
    user_data = json.loads(body)
    res = f"Received message: {user_data}"

    # Push to all connected clients
    asyncio.create_task(connected_websockets[user_data.uuid].send_text(
        res))  # async-safe call


if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("Server interrupted.")
        sys.exit(0)
