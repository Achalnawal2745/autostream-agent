import os
import time
from datetime import datetime
from langchain_core.messages import HumanMessage
from agent import create_agent
from dotenv import load_dotenv

# Initialize environment
load_dotenv()

def main():
    print("--- AutoStream AI Agent ---")
    print("Type 'exit' or 'quit' to stop.")
    
    try:
        app = create_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return
    
    # Initialize conversation state
    state = {
        "messages": [],
        "lead_details": {"name": None, "email": None, "platform": None}
    }
    
    while True:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_input = input(f"[{timestamp}] User: ")
            
            if not user_input.strip():
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
                
            state["messages"].append(HumanMessage(content=user_input))
            
            # Invoke Agent & Track Latency
            start_time = time.time()
            result = app.invoke(state)
            duration = time.time() - start_time
            
            # Sync state
            state["messages"] = result["messages"]
            state["lead_details"] = result["lead_details"]
            
            agent_response = state['messages'][-1].content
            agent_timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{agent_timestamp}] Agent (took {duration:.2f}s): {agent_response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print("\n[QUOTA ERROR] Rate limit hit. Please wait a moment.")
            else:
                print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment.")
    main()
