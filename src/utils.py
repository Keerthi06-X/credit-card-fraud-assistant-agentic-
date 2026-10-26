def get_safe_response(reason: str) -> dict:
    """
    Returns a safe fallback response for invalid inputs, negative amounts, 
    wrong feature counts, or edge cases close to thresholds.
    """
    return {
        "recommended_action": "send to human review",
        "model_rationale": f"Safe fallback triggered due to: {reason}. Review required by human agent.",
        "rule_findings": "Input flagged or score is ambiguous/near threshold.",
        "limitations": "Automated decision withheld to prevent financial risk or customer friction."
    }
