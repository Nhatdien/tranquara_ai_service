from service.rabbitmq import RabbitMQ
from langchain_core.tools import tool
from models.journal import UserJournal
from models.emotion_log import EmotionLog
from langchain_core.messages import ToolMessage
from langgraph.graph import START, END, MessagesState, StateGraph
import datetime
import json
from models.messages import *


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(
    state: MessagesState,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(
            f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


@tool("create_journal")
def create_journal(user_id: str, title: str, content: str):
    rabbitmq_conn = RabbitMQ()
    """
    Summarise the current conversation after user have sharing their current mental state and notify the user after the journal created. Write the journal
    with user perspective
    Args:
        user_id (str): user_id in uuid format
        title (str): title of the journal
        content (str): content of the journal

    Example:
        User saying they're feeling stress, you ask some gental following question before call this tool. 
    """
    journal = SyncJournalPayload(user_id=user_id, title=title, content=content)
    payload = SyncDataMessage[SyncJournalPayload](
        event="user_journal.create", payload=journal)

    rabbitmq_conn.publish("sync_data", payload.model_dump_json())
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
