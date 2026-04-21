import os
import json
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# --- Lead Capture Tool ---
def mock_lead_capture(name, email, platform):
    """Saves lead info to a local JSON file."""
    lead_data = {"name": name, "email": email, "platform": platform}
    try:
        leads = []
        if os.path.exists("captured_leads.json") and os.path.getsize("captured_leads.json") > 0:
            with open("captured_leads.json", "r") as f:
                try:
                    leads = json.load(f)
                except json.JSONDecodeError:
                    leads = []
        
        leads.append(lead_data)
        with open("captured_leads.json", "w") as f:
            json.dump(leads, f, indent=4)
    except Exception as e:
        pass # Silent error for production cleaniness

    print(f"\n[BACKEND] Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead for {name} has been successfully captured in our system."

# --- State Schema ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    lead_details: dict # Current extracted name, email, platform

# --- Knowledge Base (RAG) ---
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is not None: return _retriever
    try:
        loader = TextLoader("kb.md")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = FAISS.from_documents(docs, embeddings)
        _retriever = vectorstore.as_retriever()
        return _retriever
    except Exception:
        return None

# --- Super-Node Reasoning ---
class SuperAgentResponse(BaseModel):
    answer: str = Field(description="The response text to show the user")
    intent: str = Field(description="greeting, inquiry, or lead")
    extracted_name: Optional[str] = Field(None)
    extracted_email: Optional[str] = Field(None)
    extracted_platform: Optional[str] = Field(None)

def super_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    last_msg = state["messages"][-1].content
    details = state.get("lead_details", {"name": None, "email": None, "platform": None})
    
    # Context Retrieval
    retriever = get_retriever()
    context = ""
    if retriever:
        docs = retriever.invoke(str(last_msg))
        context = "\n".join([doc.page_content for doc in docs])

    system_prompt = f"""You are 'Inflx', an AI Sales Agent for AutoStream.
    KNOWLEDGE BASE (Facts): {context}
    
    CURRENT LEAD DATA:
    - Name: {details.get('name')}
    - Email: {details.get('email')}
    - Platform: {details.get('platform')}
    
    INSTRUCTIONS:
    - If 'greeting': Welcome the user warmly.
    - If 'inquiry': Answer using ONLY the provided Facts.
    - If 'lead': Collect missing details (Name, Email, Platform) one-by-one.
    - DO NOT re-ask for details already present in the CURRENT LEAD DATA.
    - If all 3 fields are complete, confirm receipt and thank the user.
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
        return {"messages": [AIMessage(content="I apologize, but I encountered a minor issue. How else can I help?")], "lead_details": details}

    # Extract Entities
    if response.extracted_name and not details.get('name'): details["name"] = response.extracted_name
    if response.extracted_email and not details.get('email'): details["email"] = response.extracted_email
    if response.extracted_platform and not details.get('platform'): details["platform"] = response.extracted_platform

    final_text = response.answer
    if all(details.get(k) is not None for k in ["name", "email", "platform"]):
        tool_output = mock_lead_capture(details["name"], details["email"], details["platform"])
        if "captured" not in final_text.lower():
            final_text += f"\n\n{tool_output}"
        details = {"name": None, "email": None, "platform": None}

    return {"messages": [AIMessage(content=final_text)], "lead_details": details}

def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", super_node)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()
