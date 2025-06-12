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
from models.user import UserDataForGuidence, AIGuidanceResponse
from service.rabbitmq import rabbitmq_conn


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
    ai_processor = AIProcessor()
    try:
        while True:
            # Can be replaced with actual chat triggers
            data = await websocket.receive_text()
            response = ai_processor.response_chat(data)
            print(response)
            await connected_websockets[user_uuid].send_text(response.content)
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
