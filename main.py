from service.ai_service_processor import AIProcessor
from langchain_core.output_parsers import JsonOutputParser
from fastapi import FastAPI
from models.user import UserDataForGuidence, AIGuidanceResponse
from utils.utils import init_mongo
from config import settings as global_settings
import uvicorn

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo_client, app.state.mongo_db, app.state.mongo_collection = await init_mongo(
    global_settings.MONGODB_DATABASE,
    global_settings.mongodb_url.unicode_string(),
    global_settings.MONGODB_COLLECTION,
)
    try:
        yield
    finally:
        pass

app = FastAPI(lifespan=lifespan)
@app.get("/")
async def health_check():
    print(app.state.mongo_db)
    return "app is running"

@app.post("/guidance")
async def provide_guidence(user_data: UserDataForGuidence):
    processor = AIProcessor()
    res = processor.provide_guidence_process(user_data=user_data, parser=JsonOutputParser(pydantic_object=AIGuidanceResponse))

    return res

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)