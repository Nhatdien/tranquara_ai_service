import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from database.vector_database import qdrant
from service.ai_tools import *

# Load environment variables
load_dotenv()

# Create Vector Store (FAISS)
FAISS_INDEX_PATH = "faiss_index"


class AIProcessor():
    def __init__(self, user_id: str):
        self.model = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'], model="gpt-4o-mini", streaming=True).bind_tools(
            tools=[create_journal, get_emotion_data, change_user_emotion_theme], tool_choice="auto")
        self.vector_store = qdrant

        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
        self.chat_history = []
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)

    # Define the function that calls the model

    def call_model(self, state: MessagesState):
        file = open("/app/service/test_system_prompt.md", "r")
        system_prompt = file.read()
        system_message = SystemMessage(content=system_prompt)
        # exclude the most recent user input
        message_history = state["messages"][:-1]
        # Summarize the messages if the chat history reaches a certain size
        if len(message_history) >= 5:
            print("generating summary")
            last_human_message = state["messages"][-1]
            # Invoke the model to generate conversation summary
            summary_prompt = (
                "Distill the above chat messages into a single summary message. "
                "Include as many specific details as you can."
            )
            summary_message = self.model.invoke(
                message_history + [HumanMessage(content=summary_prompt)]
            )

            # Store summarise message in the vector store

            # Delete messages that we no longer want to show up
            delete_messages = [RemoveMessage(id=m.id)
                               for m in state["messages"]]
            # Re-add user message
            human_message = HumanMessage(content=last_human_message.content)
            # Call the model with summary & response
            response = self.model.invoke(
                [system_message, summary_message, human_message])
            self.chat_history = [summary_message,
                                 human_message, response] + delete_messages
            print(
                f"generated summary and current message list: {self.chat_history}")
        else:
            self.chat_history = self.model.invoke(
                [system_message] + state["messages"])

        return {"messages": self.chat_history}

    def response_chat(self, user_input: str):
        return self.app.invoke({
            "messages": HumanMessage(content=user_input)
        },
            config={"configurable": {"thread_id": "4"}},)["messages"][-1]
