import os
from dotenv import load_dotenv
from uuid import uuid4
from service.helper import format_system_prompt
from service.prompts import get_system_prompt
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver, PersistentDict
from database.vector_database import qdrant
# from service.ai_tools import *
from models.messages import InitConnectData, UserMessagePayload
# Load environment variables
load_dotenv()


class AIProcessor():
    def __init__(self, init_metadata: InitConnectData = None):
        # self.tools = [create_journal, get_emotion_data,
        #               change_user_emotion_theme]
        self.model = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'], model="gpt-4o-mini", streaming=True)
        # .bind_tools(
        #     tools=self.tools, tool_choice="auto")
        self.vector_store = qdrant
        self.init_metadata = init_metadata
        self.currentContext = ""

        # Setting up LangGrapth nodes only if init_metadata provided (for chat)
        if init_metadata:
            self.workflow = StateGraph(state_schema=MessagesState)

            # create model node and tool node
            self.workflow.add_node("model", self.call_model)
            # self.workflow.add_node("tools", BasicToolNode(tools=self.tools))

            # create edges
            # self.workflow.add_edge("tools", "model")
            self.workflow.add_edge(START, "model")
            # self.workflow.add_conditional_edges("model", route_tools)
            self.workflow.add_edge("model", END)
            self.chat_history = []
            memory = MemorySaver()
            self.app = self.workflow.compile(checkpointer=memory)

    # Define the function that calls the model

    def call_model(self, state: MessagesState):
        # relevant_context = self.vector_store.similarity_search(
        #     query=state["messages"][-1].content, k=8, score_threshold=0.75)
        # print(
        #     "\n".join([context.page_content for context in relevant_context]))
        system_message = SystemMessage(content=format_system_prompt(
            init_data=self.init_metadata, context=self.currentContext))
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

    def response_chat(self, user_input: UserMessagePayload):
        # perform retrival query to qdrant
        self.currentContext = user_input.current_journal

        return self.app.invoke({
            "messages": HumanMessage(content=user_input.content)
        },
            config={"configurable": {"thread_id": "4"}},)["messages"][-1]

    def generate_journal_question(self, content: str, mood_score: int, slide_prompt: str = None,
                                  slide_group_context: dict = None, current_slide_id: str = None,
                                  collection_title: str = None, direction: str = None):
        """
        Generate a single follow-up question based on journal content.
        Enhanced with full slide group context for better AI awareness.

        Args:
            content: User's current journal text
            mood_score: User's mood rating (1-10)
            slide_prompt: Current slide question/prompt
            slide_group_context: Full slide group data including all slides
            current_slide_id: ID of the current slide being worked on
            collection_title: Name of the collection (e.g., "Daily Reflection")
            direction: Reflection direction ('why', 'emotions', 'patterns', 'challenge', 'growth')
        """
        # Get system prompt from external file (with optional direction enhancement)
        system_prompt = get_system_prompt(direction)

        # Build contextual information about the slide group
        context_info = []

        if collection_title:
            context_info.append(f"Collection: {collection_title}")

        if slide_group_context:
            slide_group_title = slide_group_context.get(
                'title', 'Unknown Session')
            slide_group_desc = slide_group_context.get('description', '')
            context_info.append(f"Slide Group: {slide_group_title}")
            if slide_group_desc:
                context_info.append(f"Session Purpose: {slide_group_desc}")

            # Extract all slide prompts to show the full session flow
            slides = slide_group_context.get('slides', [])
            if slides and len(slides) > 1:
                slide_questions = []
                for idx, slide in enumerate(slides, 1):
                    slide_type = slide.get('type', 'unknown')
                    question = slide.get('question', slide.get('title', ''))

                    # Mark the current slide
                    is_current = (current_slide_id and slide.get(
                        'id') == current_slide_id)
                    marker = " [CURRENT SLIDE]" if is_current else ""

                    if question:
                        slide_questions.append(
                            f"  {idx}. [{slide_type}] {question}{marker}")

                if slide_questions:
                    context_info.append(
                        f"Full Session Flow:\n" + "\n".join(slide_questions))

        context_section = "\n".join(
            context_info) if context_info else "Free journaling session"

        # Build user prompt with enhanced context
        user_prompt = f"""Journaling Session Context:
{context_section}

Current Slide Prompt: {slide_prompt or "Free journaling"}

User's Current Writing:
{content}

User's Mood Score: {mood_score}/10

Based on the FULL CONTEXT of this journaling session and what the user is writing about in the CURRENT slide, generate ONE follow-up question that:
1. Relates specifically to what they just wrote
2. Stays aligned with the theme of this slide and the overall session
3. Helps them explore their thoughts and feelings more deeply
4. Feels natural and conversational

Generate the question now:"""

        # Call LLM directly (simpler model without streaming)
        simple_model = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False
        )

        response = simple_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        question = response.content.strip()

        # Remove quotes if LLM added them
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1]

        return question
