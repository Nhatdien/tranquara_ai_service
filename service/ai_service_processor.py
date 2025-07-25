import os
from dotenv import load_dotenv
from uuid import uuid4
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, END, MessagesState, StateGraph
# from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver, PersistentDict
from database.vector_database import qdrant
from service.ai_tools import *
from models.user import UserInformations
# Load environment variables
load_dotenv()


class AIProcessor():
    def __init__(self):
        self.tools = [create_journal, get_emotion_data,
                      change_user_emotion_theme]
        self.model = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'], model="gpt-4o-mini", streaming=True).bind_tools(
            tools=self.tools, tool_choice="auto")
        self.vector_store = qdrant

        # Settinng up system message
        # self.user_info = user_info
        with open("/app/service/test_system_prompt.md", "r") as file:
            self.system_prompt_temmplate = file.read()

        # self.system_prompt_temmplate.format(**self.user_info)

        # Setting up LangGrapth nodes
        self.workflow = StateGraph(state_schema=MessagesState)

        # create model node and tool node
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_node("tools", BasicToolNode(tools=self.tools))

        # create edges
        self.workflow.add_edge("tools", "model")
        self.workflow.add_edge(START, "model")
        self.workflow.add_conditional_edges("model", route_tools)
        self.workflow.add_edge("model", END)
        self.chat_history = []
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)

    # Define the function that calls the model

    def call_model(self, state: MessagesState):
        relevent_context = self.vector_store.similarity_search(
            query=state["messages"][-1].content, k=8, score_threshold=0.75)
        print(
            "\n".join([context.page_content for context in relevent_context]))
        system_message = SystemMessage(content=self.system_prompt_temmplate.format(
            context="\n".join([context.page_content for context in relevent_context])))
        message_history = state["messages"][:-1]
        # Summarize the messages if the chat history reaches a certain size
        print(message_history)
        if len(message_history) >= 3:
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
        # perform retrival query to qdrant

        return self.app.invoke({
            "messages": HumanMessage(content=user_input)
        },
            config={"configurable": {"thread_id": "4"}},)["messages"][-1]
