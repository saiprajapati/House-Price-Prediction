"""
train.py
========
Trains multiple regression models and returns them for comparison.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import os


MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10),
    "Lasso Regression": Lasso(alpha=1000),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def train_all(X_train, y_train) -> dict:
    """
    Train all models on training data.

    Returns
    -------
    dict: {model_name: fitted_model}
    """
    fitted = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"[✓] Trained: {name}")
    return fitted


def save_model(model, path: str) -> None:
    """Persist a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved → {path}")


def load_model(path: str):
    """Load a persisted model from disk."""
    return joblib.load(path)
