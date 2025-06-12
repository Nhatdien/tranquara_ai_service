from pydantic import BaseModel, Field

# AI guidence request classes


class UserJournal(BaseModel):
    title: str
    content: str
