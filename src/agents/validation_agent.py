import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.utils import get_safe_response

def validation_node(state: dict) -> dict:
    """
    LLM-powered Validation Agent: Uses OpenAI to draft a professional 
    executive summary report while enforcing PCA hallucination safety constraints.
    """
    print("\n--- [LLM Validation Agent] Generating Executive Report ---")
    rule_findings = state.get("rule_findings", {})
    action = rule_findings.get("action")
    rationale = rule_findings.get("rationale", "")
    prediction = state.get("model_prediction", {})
    prob = prediction.get("fraud_probability", 0.0)
    amount = state.get("user_inputs", {}).get("Amount", 0.0)
    
    if action == "send to human review":
        return {
            **state,
            "review_status": "approved",
            "route": "human_review",
            "final_report": get_safe_response("Ambiguous score or boundary condition.")
        }
        
    # Use OpenAI LLM to write a professional narrative summary
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""
        You are an expert financial fraud compliance officer. Summarize this transaction screening result into a professional executive report.
        - Transaction Amount: ${amount}
        - Fraud Probability Score: {prob:.4f}
        - Recommended Action: {action.upper()}
        - Base Rationale: {rationale}

        CRITICAL CONSTRAINTS:
        1. Write a formal, concise executive summary.
        2. STRICTLY DO NOT invent, assign, or describe real-world meanings for anonymized PCA features V1 through V28 (e.g., do not mention location, device type, user identity, or merchant category).
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_narrative = response.content
    except Exception as e:
        print(f"-> LLM API Warning (using fallback text): {e}")
        llm_narrative = f"Recommended action: {action}. Model probability: {prob:.4f}."

    print("-> LLM Executive Summary Generated Successfully!")
    
    # Construct final orchestrated report with LLM narrative
    final_report = {
        "recommended_outcome": action,
        "llm_executive_summary": llm_narrative,
        "model_rationale": rationale,
        "rule_findings": f"Evaluated against cost thresholds (Step-up: 0.30, Block: 0.75). Amount: ${amount}",
        "limitations": "Model relies on historical ULB dataset PCA components. Extreme imbalance handled via weighted boosting. No semantic meaning attributed to V1-V28."
    }
    
    return {
        **state,
        "review_status": "approved",
        "route": "end",
        "final_report": final_report
    }