"""
evaluate.py
===========
Evaluates trained models and returns a comparison DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, name: str = "") -> dict:
    """
    Compute MAE, RMSE, R², and MAPE for a single model.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return {
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2),
        "_y_pred": y_pred,  # kept for plotting, stripped before display
    }


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a sorted comparison DataFrame.
    """
    rows = []
    preds = {}
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        preds[name] = result.pop("_y_pred")
        rows.append(result)

    df = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    return df, preds


def print_report(df: pd.DataFrame) -> None:
    """Pretty-print the evaluation table."""
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON REPORT")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)
    best = df.iloc[0]["Model"]
    best_r2 = df.iloc[0]["R²"]
    print(f"\n🏆  Best model: {best}  (R² = {best_r2})\n")
