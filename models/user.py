from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class UserInformation(BaseModel):
    kyc_answers: Optional[Dict[str, Any]]
    name: Optional[str]
    age: Optional[int]
    gender: Optional[str]


class UserInformations(BaseModel):
    user_id: str
    user_information: UserInformation
    user_setting: Optional[Dict[str, Any]]
