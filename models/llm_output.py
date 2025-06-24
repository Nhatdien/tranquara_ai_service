from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class LlmResponse(BaseModel):
    message: str = Field(description="The content of the response message")
    sample_response: list[str] = Field(description="3 sample user response")


parser = PydanticOutputParser(pydantic_object=LlmResponse)
