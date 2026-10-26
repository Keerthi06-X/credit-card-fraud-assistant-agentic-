import joblib
import pandas as pd
from src.config import MODEL_PATH, SCALER_PATH

def load_fraud_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_fraud(transaction_data: dict) -> dict:
    """
    Tool function called by the Fraud Scoring Agent.
    Accepts transaction data (dictionary of 30 values), scales Time and Amount,
    and returns the predicted probability and confidence score.
    """
    model, scaler = load_fraud_model()
    
    # Convert dict to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Ensure correct columns
    expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    # If Class is present in input row, drop it
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])
        
    df = df[expected_cols]
    
    # Scale Time and Amount using the saved scaler
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    
    # Predict probability of fraud (class 1)
    fraud_prob = float(model.predict_proba(df)[0][1])
    predicted_label = int(fraud_prob >= 0.5)
    
    return {
        "fraud_probability": fraud_prob,
        "predicted_label": predicted_label
    }
