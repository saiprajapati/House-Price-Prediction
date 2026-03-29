"""
preprocess.py
=============
Handles all data loading, cleaning, encoding, and splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BINARY_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]
CAT_COLS = ["furnishingstatus"]
TARGET = "price"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset and return a DataFrame."""
    df = pd.read_csv(path)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary yes/no columns to 0/1 and
    one-hot encode categorical columns.
    """
    df = df.copy()

    for col in BINARY_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
    return df


def get_features_target(df: pd.DataFrame):
    """Split DataFrame into features (X) and target (y)."""
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return X, y


def preprocess(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_save_path: str = None,
):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    """
    df = load_data(data_path)
    df = encode_features(df)
    X, y = get_features_target(df)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"[INFO] Scaler saved to {scaler_save_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
