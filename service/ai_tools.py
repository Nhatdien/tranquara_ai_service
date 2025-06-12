from service.rabbitmq import rabbitmq_conn
from langchain_core.tools import tool
from models.journal import UserJournal
from models.emotion_log import EmotionLog
import datetime
import json
from models.messages import *


@tool("create_journal", parse_docstring=True)
def create_journal(title: str, content: str):
    """
    Summarise the current conversation after user have sharing their current mental state and notify the user after the journal created

    Args:
        title (str): title of the journal
        content (str): content of the journal

    Example:
        User saying they're feeling stress, you ask some gental following question before call this tool. 
    """
    journal = UserJournal(title, content)
    payload = SyncDataMessage[SyncJournalPayload](
        event="user_journal.create", payload=journal)

    rabbitmq_conn.publish("sync_data", json.dumps(payload))
    return "Journal created"


@tool("change_user_emotion_theme")
def change_user_emotion_theme(emotion: str, source: str, context: str):
    """
    Change user's current emotion state from the last log entry

    Args:
        emotion (str): the last emotion log entry user created
        source (str): "ai"
        context (str): summarize the context of the conversation that the emotion log is created

    Example:
        User saying they're feeling stress, you ask some gental following question before call this tool. .
    """
    # Optional: Convert timestamp to datetime
    emotion_log = EmotionLog(emotion=emotion, source=source, context=context)
    print(emotion_log)
    return "Emotion theme changed"


@tool("get_emotion_data")
def get_emotion_data(start_time: str, end_time: str):
    """
    Get the emotion log in some period of time. Provide ISO 8601 datetime strings.
    """
    start = datetime.datetime.fromisoformat(start_time)
    end = datetime.datetime.fromisoformat(end_time)
    return f"Querying logs from {start} to {end}"
