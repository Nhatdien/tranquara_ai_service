from pydantic import BaseModel, Field

# AI guidence request classes


class EmotionLog(BaseModel):
    emotion: str
    source: str
    context: str
