"""
train_model.py
--------------
Trains a Ridge Regression model on the Kaggle 'Home Value Insights' dataset
(house_price.csv) and saves:
  - model/house_price_model.pkl
  - report_images/feature_importance.png
  - report_images/actual_vs_predicted.png
  - report_images/residuals.png
  - report_images/price_distribution.png
  - report_images/correlation_heatmap.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "house_price.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
REPORT_DIR = os.path.join(BASE_DIR, "report_images")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Features (match Kaggle dataset columns) ────────────────────────────────────
FEATURES = [
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
    "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality",
]
TARGET = "House_Price"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    return X, y, df


def train():
    print("Loading data …")
    X, y, df = load_data()
    print(f"  {len(df)} records | {len(FEATURES)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0)),
    ])

    print("Training Ridge Regression …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

    print(f"\n{'='*40}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"{'='*40}\n")

    # ── Save model ────────────────────────────────────────────────────────────
    meta = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target":   TARGET,
        "metrics":  {
            "r2": r2, "mae": mae, "rmse": rmse,
            "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std()),
        },
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }
    joblib.dump(meta, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.05)
    BLUE   = "#3b82d4"
    PURPLE = "#7c5cd8"

    # 1. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test / 1e6, y_pred / 1e6, alpha=0.45, edgecolors="none",
               color=BLUE, s=25, label="Predictions")
    lims = [min(y_test.min(), y_pred.min()) / 1e6,
            max(y_test.max(), y_pred.max()) / 1e6]
    ax.plot(lims, lims, "r--", linewidth=1.8, label="Perfect fit")
    ax.set_xlabel("Actual Price ($M)")
    ax.set_ylabel("Predicted Price ($M)")
    ax.set_title("Actual vs Predicted House Prices")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "actual_vs_predicted.png"), dpi=130)
    plt.close(fig)
    print("  ✓ actual_vs_predicted.png")

    # 2. Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred / 1e6, residuals / 1e3, alpha=0.45, edgecolors="none",
               color=PURPLE, s=25)
    ax.axhline(0, color="red", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Predicted Price ($M)")
    ax.set_ylabel("Residual ($K)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "residuals.png"), dpi=130)
    plt.close(fig)
    print("  ✓ residuals.png")

    # 3. Feature Importance (|Ridge coefficients| after scaling)
    coefs    = pipeline.named_steps["model"].coef_
    feat_imp = pd.Series(np.abs(coefs), index=FEATURES).sort_values(ascending=True)
    fig, ax  = plt.subplots(figsize=(7, 5))
    feat_imp.plot(kind="barh", ax=ax, color=BLUE, edgecolor="white")
    ax.set_xlabel("|Coefficient| (scaled units)")
    ax.set_title("Feature Importance — Ridge |Coef|")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=130)
    plt.close(fig)
    print("  ✓ feature_importance.png")

    # 4. Price Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET] / 1e6, bins=50, kde=True, ax=ax, color=BLUE)
    ax.set_xlabel("House Price ($M)")
    ax.set_title("House Price Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "price_distribution.png"), dpi=130)
    plt.close(fig)
    print("  ✓ price_distribution.png")

    # 5. Correlation Heatmap
    corr = df[FEATURES + [TARGET]].corr().round(2)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "correlation_heatmap.png"), dpi=130)
    plt.close(fig)
    print("  ✓ correlation_heatmap.png")

    print("\nAll report images saved → report_images/")
    return meta


if __name__ == "__main__":
    train()
