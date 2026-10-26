from src.model_tool import predict_fraud

def fraud_scoring_node(state: dict) -> dict:
    """
    Fraud Scoring Agent: Calls the trained model tool and returns the fraud probability.
    """
    print("\n--- [Fraud Scoring Agent] Calling Model Tool ---")
    user_input = state.get("user_inputs", {})
    
    # Call the model tool strictly (never LLM guess)
    prediction = predict_fraud(user_input)
    print(f"-> Model Prediction Result: {prediction}")
    
    return {
        **state,
        "model_prediction": prediction,
        "route": "policy"
    }