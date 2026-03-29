"""
visualize.py
============
All plotting routines for EDA and model evaluation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = "plots"
PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette=PALETTE)


def _save(fig, filename: str) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {path}")
    return path


# ── EDA ──────────────────────────────────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("House Price Distribution", fontsize=14, fontweight="bold")

    sns.histplot(df["price"], kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Distribution")
    axes[0].set_xlabel("Price (₹)")

    sns.boxplot(x=df["price"], ax=axes[1], color="#4C72B0")
    axes[1].set_title("Box Plot")
    axes[1].set_xlabel("Price (₹)")

    fig.tight_layout()
    return _save(fig, "01_price_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_correlation_heatmap.png")


def plot_feature_vs_price(df: pd.DataFrame) -> str:
    num_cols = [
        c for c in ["area", "bedrooms", "bathrooms", "stories", "parking"]
        if c in df.columns
    ]
    fig, axes = plt.subplots(1, len(num_cols), figsize=(5 * len(num_cols), 4))
    fig.suptitle("Numeric Features vs Price", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, num_cols):
        sns.scatterplot(x=df[col], y=df["price"], ax=ax, alpha=0.6, color="#4C72B0")
        ax.set_xlabel(col.title())
        ax.set_ylabel("Price (₹)")

    fig.tight_layout()
    return _save(fig, "03_features_vs_price.png")


def plot_categorical_impact(df: pd.DataFrame) -> str:
    binary_cols = [
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning", "prefarea",
    ]
    binary_cols = [c for c in binary_cols if c in df.columns]
    n = len(binary_cols)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle("Binary Feature Impact on Price", fontsize=14, fontweight="bold")

    for i, col in enumerate(binary_cols):
        ax = axes[i]
        plot_df = df.copy()
        plot_df[col] = plot_df[col].map({"yes": "Yes", "no": "No"})
        sns.boxplot(x=col, y="price", hue=col, data=plot_df, ax=ax,
                    palette=["#4C72B0", "#DD8452"], legend=False)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Price (₹)" if i % 3 == 0 else "")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return _save(fig, "04_categorical_impact.png")


# ── MODEL EVALUATION ─────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> str:
    metrics = ["R²", "MAE", "RMSE", "MAPE (%)"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

    for ax, metric in zip(axes.flatten(), metrics):
        sorted_df = results_df.sort_values(metric, ascending=(metric != "R²"))
        colors = ["#2ecc71" if i == 0 else "#4C72B0" for i in range(len(sorted_df))]
        bars = ax.barh(sorted_df["Model"], sorted_df[metric], color=colors)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel(metric)
        for bar in bars:
            w = bar.get_width()
            ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                    f"{w:,.2f}", va="center", fontsize=8)

    fig.tight_layout()
    return _save(fig, "05_model_comparison.png")


def plot_actual_vs_predicted(y_test, preds: dict, best_model_name: str) -> str:
    y_pred = preds[best_model_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Actual vs Predicted — {best_model_name}", fontsize=13, fontweight="bold")

    # Scatter
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolors="white", s=50)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Scatter Plot")
    ax.legend()

    # Residuals
    ax2 = axes[1]
    residuals = y_test.values - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color="#DD8452", edgecolors="white", s=50)
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted Price (₹)")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot")

    fig.tight_layout()
    return _save(fig, "06_actual_vs_predicted.png")


def plot_feature_importance(model, feature_names: list, model_name: str) -> str:
    if not hasattr(model, "feature_importances_"):
        return None

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return _save(fig, "07_feature_importance.png")


def plot_all_predictions(y_test, preds: dict) -> str:
    n = len(preds)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    fig.suptitle("Predicted vs Actual for All Models", fontsize=14, fontweight="bold")

    for i, (name, y_pred) in enumerate(preds.items()):
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.55, s=30, color="#4C72B0")
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lim, lim, "r--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return _save(fig, "08_all_model_predictions.png")
