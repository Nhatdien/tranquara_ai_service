from models.user import UserInformations
import requests
import os


def get_user_information(token: str) -> requests.Response:
    return requests.get(f"{os.getenv("CORE_SERVICE_URL")}/user_information", headers={
        "Authorization": token
    })
