from src.utils import get_safe_response

def intake_node(state: dict) -> dict:
    """
    Transaction Intake Agent: Accepts a transaction (JSON dict of values) 
    and validates it. Checks feature count and negative amounts.
    """
    print("\n--- [Intake Agent] Validating Transaction ---")
    user_input = state.get("user_inputs", {})
    
    # Check for negative Amount
    if "Amount" in user_input and float(user_input["Amount"]) < 0:
        print("-> Validation Failed: Negative Amount detected!")
        return {**state, "route": "human_review", "final_report": get_safe_response("Negative transaction amount detected.")}
        
    # Check feature count (Time + 28 V components + Amount = 30 features)
    expected_keys = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing_keys = [k for k in expected_keys if k not in user_input]
    
    if len(missing_keys) > 0:
        print(f"-> Validation Failed: Missing features {missing_keys}")
        return {**state, "route": "human_review", "final_report": get_safe_response(f"Missing required features: {missing_keys}")}
        
    print("-> Input successfully validated. Routing to Fraud Scoring Agent.")
    return {**state, "route": "fraud"}