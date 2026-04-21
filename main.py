import sys
import os
import time
from datetime import datetime
from langchain_core.messages import HumanMessage
from agent import create_agent
from dotenv import load_dotenv

load_dotenv()

def main():
    print("--- AutoStream AI Agent (Super-Node Optimized) ---")
    print("Type 'exit' or 'quit' to stop.")
    
    try:
        app = create_agent()
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return
    
    # Simple state
    state = {
        "messages": [],
        "lead_details": {"name": None, "email": None, "platform": None}
    }
    
    while True:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_input = input(f"\n[{timestamp}] User: ")
            
            if not user_input.strip():
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
                
            # Add user message
            state["messages"].append(HumanMessage(content=user_input))
            
            # Measure performance
            start_time = time.time()
            result = app.invoke(state)
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Update state with result
            state["messages"] = result["messages"]
            state["lead_details"] = result["lead_details"]
            
            # Clean output display
            agent_response = state['messages'][-1].content
            if isinstance(agent_response, list):
                agent_response = "".join([p['text'] if isinstance(p, dict) else p for p in agent_response])
                
            agent_timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{agent_timestamp}] Agent (took {duration:.2f}s): {agent_response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            # Check for API-specific errors (likely 429)
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg:
                print("\n[QUOTA ERROR] You've hit the Gemini free tier limit. Please wait 1 minute.")
            else:
                print(f"\n[ERROR] {err_msg}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("CRITICAL: GOOGLE_API_KEY is missing in .env")
    main()
