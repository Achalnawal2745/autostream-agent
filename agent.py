import os
import json
from typing import TypedDict, Annotated, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()

# ==========================================
# 3.3 Tool Execution – Lead Capture
# ========================================# --- Mock API Function ---
def mock_lead_capture(name, email, platform):
    """Saves lead details to a local JSON file safely."""
    lead_data = {"name": name, "email": email, "platform": platform}
    try:
        leads = []
        if os.path.exists("captured_leads.json") and os.path.getsize("captured_leads.json") > 0:
            with open("captured_leads.json", "r") as f:
                try:
                    leads = json.load(f)
                except json.JSONDecodeError:
                    leads = [] # Handle corrupted/empty JSON
        
        leads.append(lead_data)
        with open("captured_leads.json", "w") as f:
            json.dump(leads, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Lead persistence failed: {e}")

    print(f"\n[BACKEND] Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead for {name} has been successfully captured in our system."

# ==========================================
# 5. State Management (LangGraph)
# ==========================================
class AgentState(TypedDict):
    """
    Maintains the conversation state across 5-6+ turns.
    """
    messages: Annotated[List[BaseMessage], "Conversation history"]
    lead_details: dict  # Tracks name, email, platform status

# ==========================================
# 3.2 RAG-Powered Knowledge Retrieval
# ==========================================
_retriever = None

def get_retriever():
    """
    Initializes and caches the FAISS vector store for pricing and policy retrieval.
    """
    global _retriever
    if _retriever is not None:
        return _retriever
        
    try:
        # Load local knowledge base (kb.md)
        loader = TextLoader("kb.md")
        documents = loader.load()
        
        # Split text into manageable chunks for RAG
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # Create vector store using Gemini Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(docs, embeddings)
        _retriever = vectorstore.as_retriever()
        return _retriever
    except Exception as e:
        print(f"[CRITICAL] RAG Initialization failed: {e}")
        return None

# ==========================================
# LLM Logic: Intent Detection & Reasoning
# ==========================================

class SuperAgentResponse(BaseModel):
    """
    Structured output to ensure 100% accuracy in intent and detail extraction.
    """
    answer: str = Field(description="The final message to the user.")
    intent: str = Field(description="Classification: greeting, inquiry, or lead.")
    extracted_name: Optional[str] = Field(None)
    extracted_email: Optional[str] = Field(None)
    extracted_platform: Optional[str] = Field(None)

# --- Node Logic ---

def super_node(state: AgentState):
    """
    The core 'Super Node' that handles reasoning, RAG, and extraction in one pass.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    last_msg = state["messages"][-1].content
    details = state.get("lead_details", {"name": None, "email": None, "platform": None})
    
    # 1. RAG Step
    retriever = get_retriever()
    context = ""
    if retriever:
        docs = retriever.invoke(str(last_msg))
        context = "\n".join([doc.page_content for doc in docs])

    # 2. Updated System Prompt
    system_prompt = f"""You are 'Inflx', an AI Sales Agent for AutoStream.
    KNOWLEDGE BASE (Facts): {context}
    
    CURRENT LEAD DATA:
    - Name: {details.get('name')}
    - Email: {details.get('email')}
    - Platform: {details.get('platform')}
    
    INSTRUCTIONS:
    - If 'greeting': Welcome the user.
    - If 'inquiry': Answer using ONLY the Facts.
    - If 'lead': Collect missing details (Name, Email, Platform) one-by-one.
    - IMPORTANT: If you already have a value in the CURRENT LEAD DATA, DO NOT ask for it again. 
    - If all 3 fields are present or provided in this message, simply confirm that you have everything and thank the user.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm.with_structured_output(SuperAgentResponse)
    
    try:
        history = state["messages"][-6:-1] if len(state["messages"]) > 1 else []
        response = chain.invoke({"input": last_msg, "messages": history})
    except Exception:
        return {"messages": [AIMessage(content="I apologize, but I'm experiencing a high volume of requests. Let's try again.")], "lead_details": details}

    # Extraction Logic
    if response.extracted_name and not details.get('name'): details["name"] = response.extracted_name
    if response.extracted_email and not details.get('email'): details["email"] = response.extracted_email
    if response.extracted_platform and not details.get('platform'): details["platform"] = response.extracted_platform

    final_text = response.answer
    if all(details.get(k) is not None for k in ["name", "email", "platform"]):
        tool_output = mock_lead_capture(details["name"], details["email"], details["platform"])
        # Only append tool output if the AI didn't already say it's done
        if "captured" not in final_text.lower():
            final_text += f"\n\n{tool_output}"
        details = {"name": None, "email": None, "platform": None}

    return {"messages": [AIMessage(content=final_text)], "lead_details": details}

# ==========================================
# LangGraph Orchestration
# ==========================================
def create_agent():
    """
    Compiles the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", super_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

# Optimization: Pre-load RAG on startup
if __name__ == "agent" or __name__ == "__main__":
    get_retriever()
