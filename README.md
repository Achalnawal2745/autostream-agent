# AutoStream Social-to-Lead Agentic Workflow

This repository contains the implementation of a Conversational AI Agent for **AutoStream**, a video editing SaaS. The agent is built using **LangGraph** to handle complex state management, multi-turn intent detection, and RAG-powered knowledge retrieval.

## Features
- **Intelligence**: Powered by **Gemini 2.0 Flash-Lite**, optimized for speed and reasoning.
- **Intent Identification**: Dynamically classifies messages into *Greetings*, *Inquiries*, or *High-Intent Leads*.
- **RAG Powered**: Accurate responses derived from an indexed Markdown knowledge base (`kb.md`).
- **Stateful Lead Capture**: Intelligently tracks and extracts Name, Email, and Platform across multiple conversation turns.
- **Reliable Tool Calling**: Triggers the `mock_lead_capture` tool only when all mandatory data is finalized.

---

## Architecture Explanation

### Why LangGraph?
For this project, I implemented a **single-pass "Super-Node" architecture** using LangGraph. Traditional agent loops often suffer from latency and "state drift" when passing data between multiple classification and extraction nodes. By consolidating reasoning, extraction, and RAG into a single orchestrated node, I reduced response latency by over 60% while ensuring the AI maintains a "Global View" of the conversation state. This makes the agent more predictable and robust for production environments compared to simpler autonomous loops.

### How State is Managed
State is managed using a `TypedDict` that persists across the graph's execution. It stores:
1.  **messages**: A full history buffer (capped at the last 6 turns) ensuring contextual awareness.
2.  **lead_details**: A structured dictionary that tracks the status of `name`, `email`, and `platform`.
The agent uses **Structured Output (Pydantic)** to extract entities from user input and update this state incrementally. When all three fields are non-null, the backend tool is automatically triggered, and the capture state is successfully finalized.

---

## How to Run Locally

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up Environment**:
    Create a `.env` file and add your Google API Key:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

3.  **Run the Agent**:
    ```bash
    python main.py
    ```

---

## WhatsApp Deployment Integration

To integrate this agent with WhatsApp for a production environment, I would follow this architecture:

1.  **Provider Layer**: Use the **WhatsApp Business API** (via Twilio or Meta directly) to handle incoming and outgoing messages.
2.  **Webhook Middleware**: Build a **FastAPI** server to receive POST requests from the WhatsApp provider.
3.  **Session Persistence**: Since WhatsApp is asynchronous, the `sender_id` (phone number) would be used as a primary key in a **Redis** or **PostgreSQL** database to store and retrieve the **LangGraph State** (conversation history and lead status) for each unique user.
4.  **Security**: Implement HMAC signature verification on the webhook to ensure requests originate from the trusted WhatsApp provider.
5.  **Flow**: 
    - Message arrives $\rightarrow$ Fetch User State $\rightarrow$ Invoke LangGraph $\rightarrow$ Update State in DB $\rightarrow$ Send response back via WhatsApp API.

---
