import pandas as pd
from src.agents.graph import create_fraud_screening_graph
from src.config import DATA_PATH

def run_screening_demo():
    print("==================================================")
    print(" CREDIT CARD TRANSACTION SCREENING ASSISTANT")
    print("==================================================")
    
    # 1. Load a test transaction from the dataset (e.g., from the test set)
    df = pd.read_csv(DATA_PATH)
    
    # Pick a sample transaction (let's pick one known fraud or normal transaction)
    # For demonstration, let's grab the first row or a sample row
    sample_row = df.iloc[100].to_dict()
    
    print(f"\n[Test Input Loaded] Transaction ID row #100")
    print(f"Amount: ${sample_row.get('Amount')}, Time: {sample_row.get('Time')}")
    
    # 2. Initialize Shared State dictionary
    initial_state = {
        "user_inputs": sample_row,
        "route": "intake",
        "model_prediction": None,
        "rule_findings": None,
        "draft": None,
        "review_status": "pending",
        "retry_count": 0,
        "final_report": None
    }
    
    # 3. Build and compile LangGraph workflow
    app = create_fraud_screening_graph()
    
    print("\n--- Executing Multi-Agent Workflow ---")
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n==================================================")
    print(" FINAL ORCHESTRATED ADVISORY REPORT")
    print("==================================================")
    report = final_state.get("final_report", {})
    for key, value in report.items():
        print(f"• {key.replace('_', ' ').title()}: {value}")
    print("==================================================")

if __name__ == "__main__":
    run_screening_demo()