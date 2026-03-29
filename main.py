"""
main.py
=======
Orchestrates the full ML pipeline:
  1. Preprocess data
  2. Train all models
  3. Evaluate and compare
  4. Generate all plots
  5. Save best model + scaler
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess
from src.train import train_all, save_model
from src.evaluate import evaluate_all, print_report
from src.visualize import (
    plot_price_distribution,
    plot_correlation_heatmap,
    plot_feature_vs_price,
    plot_categorical_impact,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_all_predictions,
)
import pandas as pd


DATA_PATH = "data/house_data.csv"
SCALER_PATH = "models/scaler.pkl"
BEST_MODEL_PATH = "models/best_model.pkl"
PLOTS_DIR = "plots"


def main():
    print("\n" + "=" * 55)
    print("  🏠  HOUSE PRICE PREDICTOR — ML PIPELINE")
    print("=" * 55)

    # ── 1. EDA Plots ─────────────────────────────────────────
    print("\n[1/5] Generating EDA plots...")
    raw_df = pd.read_csv(DATA_PATH)
    plot_price_distribution(raw_df)
    plot_correlation_heatmap(raw_df)
    plot_feature_vs_price(raw_df)
    plot_categorical_impact(raw_df)

    # ── 2. Preprocessing ─────────────────────────────────────
    print("\n[2/5] Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
        DATA_PATH, scaler_save_path=SCALER_PATH
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Features: {feature_names}")

    # ── 3. Training ──────────────────────────────────────────
    print("\n[3/5] Training models...")
    models = train_all(X_train, y_train)

    # ── 4. Evaluation ────────────────────────────────────────
    print("\n[4/5] Evaluating models...")
    results_df, preds = evaluate_all(models, X_test, y_test)
    print_report(results_df)

    # Save results CSV
    os.makedirs("models", exist_ok=True)
    results_df.to_csv("models/model_results.csv", index=False)

    # ── 5. Plots + Save Best ──────────────────────────────────
    print("[5/5] Generating evaluation plots & saving best model...")
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    plot_model_comparison(results_df)
    plot_actual_vs_predicted(y_test, preds, best_model_name)
    plot_feature_importance(best_model, feature_names, best_model_name)
    plot_all_predictions(y_test, preds)

    save_model(best_model, BEST_MODEL_PATH)

    print("\n" + "=" * 55)
    print(f"  ✅  Pipeline complete!")
    print(f"  🏆  Best Model : {best_model_name}")
    print(f"  📊  R²         : {results_df.iloc[0]['R²']}")
    print(f"  📉  MAE        : ₹{results_df.iloc[0]['MAE']:,.0f}")
    print(f"  💾  Model saved: {BEST_MODEL_PATH}")
    print(f"  🖼️   Plots saved: {PLOTS_DIR}/")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
