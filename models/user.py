from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class UserInformation(BaseModel):
    kyc_answers: Optional[Dict[str, Any]]
    name: Optional[str]
    age_range: Optional[str]
    gender: Optional[str]


class UserInformations(UserInformation):
    user_id: str
    user_setting: Optional[Dict[str, Any]] = None
