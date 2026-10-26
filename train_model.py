import os
import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report
from xgboost import XGBClassifier
from src.config import DATA_PATH, MODEL_PATH, SCALER_PATH
from src.preprocessing import load_and_inspect_data, get_time_based_splits

def train_and_evaluate():
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    
    # 1. Load and inspect data
    df = load_and_inspect_data(DATA_PATH)
    
    # 2. Get time-based splits (train on earlier, test on later)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = get_time_based_splits(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 3. Train a classifier suited for extreme imbalance (XGBoost)
    print("\nTraining XGBoost model...")
    # Scale pos weight handles the extreme class imbalance (0.172% fraud)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr'
    )
    
    model.fit(X_train, y_train)
    
    # 4. Evaluate on Validation set (PR-AUC, Confusion Matrix as required)
    val_probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    pr_auc = auc(recall, precision)
    
    print(f"\n--- Validation Metrics ---")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Check for overfitting (Compare train vs validation performance)
    train_probs = model.predict_proba(X_train)[:, 1]
    train_precision, train_recall, _ = precision_recall_curve(y_train, train_probs)
    train_pr_auc = auc(train_recall, train_precision)
    print(f"Train PR-AUC: {train_pr_auc:.4f} vs Validation PR-AUC: {pr_auc:.4f} (Checking for overfitting)")
    
    # Confusion matrix at default threshold (or custom threshold)
    val_preds = (val_probs >= 0.5).astype(int)
    print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))
    
    # 5. Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
    