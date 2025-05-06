from pydantic import BaseModel, Field

# AI guidence request classes


class UserDataForGuidence(BaseModel):
    current_week: int
    chatbot_interaction: str
    emotion_tracking: str


class AIGuidanceResponse(BaseModel):
    suggest_mindfulness_tip: str = Field(description="Tip for the user")
    explaination: str = Field(description="Explanation for the tip provided")
