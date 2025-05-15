import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import AI_SERVICE_CONFIG as cfg
from models.user import UserDataForGuidence

# Load environment variables
load_dotenv()

# Create Vector Store (FAISS)
FAISS_INDEX_PATH = "faiss_index"


class AIProcessor():
    def __init__(self):
        self.model = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'], model="gpt-4o-mini", streaming=True)
        self.vector_store = self.load_or_create_faiss_index()
        self.retriever = self.vector_store.as_retriever()

    def provide_guidence_process(self, user_data: UserDataForGuidence, parser: JsonOutputParser):
        prompt_text = '''
            You are an AI-powered mindfulness chatbot. The user follows a **structured 8-week mindfulness program**.
            With context: {context}

            Current user week: {current_week}
            Recent Chatbot Queries: {chatbot_interaction}

            Emotion Tracking: {emotion_tracking}

            Your task:

            1. Analyze their mood trends and chatbot queries.
            2. Suggest a mindfulness tip that aligns with their **current structured program week**.

            Output format:
            {format_instruction}
         '''
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=[
                "current_week", "chatbot_interaction", "emotion_tracking", "context"],
            partial_variables={
                "format_instruction": parser.get_format_instructions()}
        )

        user_data = user_data.__dict__
        query = f"Mindfulness guidance for Week {user_data['current_week']} related to {user_data['chatbot_interaction']}"
        retrieved_docs = self.retriever.invoke(input=query)

        context_text = "\n".join(
            [page.page_content for page in retrieved_docs])
        # Define Chain for Processing
        chain = (
            prompt
            | self.model
            | parser
        )

        # Invoke Chain
        res = chain.invoke({
            "context": context_text,
            "current_week": user_data["current_week"],
            "chatbot_interaction": user_data["chatbot_interaction"],
            "emotion_tracking": user_data["emotion_tracking"]
        })

        return res

    def load_or_create_faiss_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            print("✅ FAISS index found. Loading from disk...")
            vector_store = FAISS.load_local(
                cfg["FAISS_INDEX_PATH"], OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        else:
            # Load and split PDF document
            pdf_path = os.path.abspath(cfg("DOCUMENT_PATH"))
            loader = PyPDFLoader(file_path=pdf_path)
            print("⚠️ FAISS index not found. Creating new index...")
            pages = [page for page in loader.load_and_split()]
            print(f"Loaded {len(pages)} pages.")
            vector_store = FAISS.from_documents(pages, OpenAIEmbeddings())
            # Save to disk for future use
            vector_store.save_local(FAISS_INDEX_PATH)
            print("✅ FAISS index saved for future runs.")

        return vector_store
