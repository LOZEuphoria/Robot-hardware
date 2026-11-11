import sys
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage
from huflit_rag_agent import create_agent_graph

def run_cli():
    # Force UTF-8 encoding for standard output on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=========================================================")
    print("      HUFLIT STATEFUL AI RAG AGENT CHAT CONSOLE")
    print("=========================================================")
    print("Loading Agent Graph & Local LLM...")
    
    # Compile the graph
    graph = create_agent_graph()
    
    thread_id = str(uuid.uuid4())[:8]
    print(f"\n[Hệ thống] Đã tạo phiên chat mới (Thread ID: {thread_id})")
    
    # Display help commands
    print("\nCác lệnh đặc biệt hỗ trợ:")
    print("  /new      : Bắt đầu phiên hội thoại mới (Xóa bộ nhớ ngắn hạn)")
    print("  /exit     : Thoát chương trình")
    print("---------------------------------------------------------")
    
    while True:
        try:
            user_input = input("\n[Student] > ").strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            cmd = user_input.lower().split()
            if cmd[0] == "/exit" or cmd[0] == "/quit":
                print("Tạm biệt sinh viên!")
                break
                
            elif cmd[0] == "/new":
                thread_id = str(uuid.uuid4())[:8]
                print(f"[Hệ thống] Đã đặt lại phiên chat mới. Bộ nhớ ngắn hạn trống (Thread ID: {thread_id})")
                continue
                
                
            
            # Prepare config for LangGraph (thread-level short-term checkpoint memory)
            config = {"configurable": {"thread_id": thread_id}}
            
            # Input State
            inputs = {
                "messages": [HumanMessage(content=user_input)]
            }
            
            print("Processing...", end="\r")
            
            # Run the graph synchronously
            state = graph.invoke(inputs, config)
            
            # Display detailed step routing for testing visual feedback
            rewritten = state.get("rewritten_query")
            next_step = state.get("next_step")
            docs = state.get("retrieved_documents", [])
            response = state.get("generation", "")
            
            # Prints metadata logs
            print(f"[Agent] Router: {next_step} | Rewritten: '{rewritten}' | Docs: {len(docs)}")
            print(f"Answer: {response}")
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"\n[Lỗi] Đã xảy ra sự cố: {e}")

if __name__ == "__main__":
    run_cli()
