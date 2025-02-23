from service.ai_service_processor import AIProcessor
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from fastapi import FastAPI
from models.user import UserDataForGuidence, AIGuidanceResponse
import uvicorn
# Example User Data

app = FastAPI()
@app.get("/")
async def health_check():
    return "app is running"

@app.post("/guidance")
async def provide_guidence(user_data: UserDataForGuidence):
    print(type(user_data))
    processor = AIProcessor()
    res = processor.provide_guidence_process(user_data=user_data, parser=JsonOutputParser(pydantic_object=AIGuidanceResponse))

    return res

if __name__ == "__main__":
    uvicorn.run(app="main:app", reload=True)