import os
from dotenv import load_dotenv

load_dotenv()

# Data & Model Paths
DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"

# Cost Policy Matrix Parameters
COST_FALSE_ALARM = 5.0     # Friction/step-up cost
COST_MISSED_FRAUD = 100.0  # Fraud loss cost

# Thresholds derived from PR curve & cost table (not 0.5)
THRESHOLD_STEP_UP = 0.30
THRESHOLD_BLOCK = 0.75
