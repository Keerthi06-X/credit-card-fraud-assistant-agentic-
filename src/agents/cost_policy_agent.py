from src.config import THRESHOLD_STEP_UP, THRESHOLD_BLOCK

def cost_policy_node(state: dict) -> dict:
    """
    Cost Policy Agent: Rule-based mapping of probability and Amount to an action
    using cost table thresholds (approve, step-up verification, block).
    """
    print("\n--- [Cost Policy Agent] Applying Cost-Benefit Rules ---")
    prediction = state.get("model_prediction", {})
    prob = prediction.get("fraud_probability", 0.0)
    user_input = state.get("user_inputs", {})
    amount = float(user_input.get("Amount", 0.0))
    
    # Check if score is extremely close to a threshold (triggering safe review)
    if abs(prob - THRESHOLD_STEP_UP) < 0.02 or abs(prob - THRESHOLD_BLOCK) < 0.02:
        print("-> Score is ambiguous / near threshold. Triggering human review.")
        return {
            **state,
            "route": "validation",
            "rule_findings": {
                "action": "send to human review",
                "reason": "Probability score is too close to decision threshold."
            }
        }
    if prob >= THRESHOLD_BLOCK:
        action = "block"
        rationale = f"High fraud probability ({prob:.2f}) exceeds block threshold ({THRESHOLD_BLOCK})."
    elif prob >= THRESHOLD_STEP_UP:
        action = "step-up verification"
        rationale = f"Moderate fraud probability ({prob:.2f}) exceeds step-up threshold ({THRESHOLD_STEP_UP})."
    else:
        action = "approve"
        rationale = f"Low fraud probability ({prob:.2f}). Safe to approve."
        
    rule_findings = {
        "action": action,
        "rationale": rationale,
        "probability": prob,
        "amount": amount
    }
    
    print(f"-> Policy Decision: {action.upper()} | Rationale: {rationale}")
    
    return {
        **state,
        "rule_findings": rule_findings,
        "route": "validation"
    }