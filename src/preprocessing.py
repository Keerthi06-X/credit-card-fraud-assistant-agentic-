import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_inspect_data(filepath):
    """Loads dataset and inspects missing values, duplicates, and label distribution."""
    df = pd.read_csv(filepath)
    print("--- Dataset Inspection ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("Class distribution:\n", df['Class'].value_counts(normalize=True))
    return df

def get_time_based_splits(df):
    """
    Splits data by time (earlier transactions for training, later for testing).
    Why random split overstates performance: 
    Credit card fraud patterns evolve or have temporal correlations; a random split 
    leaks future transaction patterns into the past, causing data leakage and overly optimistic metrics.
    """
    # Sort by Time to ensure chronological order
    df = df.sort_values('Time').reset_index(drop=True)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # Time-based split: 70% train, 15% validation, 15% test
    train_idx = int(len(df) * 0.70)
    val_idx = int(len(df) * 0.85)
    
    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
    
    # Fit scaler strictly on training data only
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_val[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler