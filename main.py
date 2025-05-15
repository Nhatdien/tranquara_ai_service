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

dotenv.load_dotenv()


# --- RabbitMQ Consumer Task ---
async def start_rabbitmq_consumer():
    rabbitmq = RabbitMQ()
    rabbitmq.channel.queue_declare("ai_tasks")
    rabbitmq.channel.queue_declare("ai_response")

    loop = asyncio.get_event_loop()

    def run():
        try:
            print("Starting RabbitMQ consumer...")
            rabbitmq.consume(queue_name='ai_tasks', callback=rabbitmq_callback)
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

    yield  # App is now running

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
connected_websockets: List[WebSocket] = []

# --- WebSocket Route ---


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # Can be replaced with actual chat triggers
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
        print("WebSocket disconnected")


# --- RabbitMQ Callback Handler ---
def rabbitmq_callback(ch, method, properties, body):
    ai_processor = AIProcessor()
    parser = JsonOutputParser(pydantic_object=AIGuidanceResponse)

    user_data = json.loads(body)
    user_pass_data = UserDataForGuidence(**user_data)
    res = ai_processor.provide_guidence_process(
        user_data=user_pass_data, parser=parser)

    ch.basic_publish(
        exchange='',
        routing_key="ai_response",
        body=json.dumps(res),
        properties=pika.BasicProperties(delivery_mode=2)
    )

    print("AI result:", res)

    # Push to all connected clients
    for ws in connected_websockets:
        asyncio.create_task(ws.send_text(json.dumps(res)))  # async-safe call


# --- Launch App with Uvicorn ---
if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
    except KeyboardInterrupt:
        print("Server interrupted.")
        sys.exit(0)
