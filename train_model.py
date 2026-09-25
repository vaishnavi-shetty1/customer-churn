from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "customer_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
COLORS = {
    "Retained": "#0f766e",
    "Churned": "#e11d48",
    "accent": "#0f766e",
    "secondary": "#fb7185",
    "grid": "#e2e8f0",
    "text": "#0f172a",
}


def apply_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": COLORS["grid"],
            "font.size": 10,
        }
    )


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = pd.read_csv(path)
    original_rows = len(data)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    missing_total_charges = int(data["TotalCharges"].isna().sum())
    data = data.loc[data["TotalCharges"].notna()].copy()
    data = data.drop(columns=["customerID"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    if data.isna().any().any():
        raise ValueError("The cleaned dataset still contains missing values.")
    if not set(data["Churn"].unique()).issubset({0, 1}):
        raise ValueError("The Churn target contains unexpected values.")
    if len(data) != original_rows - missing_total_charges:
        raise ValueError("Unexpected row count after cleaning TotalCharges.")
    cleaning_summary = {
        "original_rows": original_rows,
        "cleaned_rows": len(data),
        "removed_missing_total_charges": missing_total_charges,
        "duplicate_rows": int(data.duplicated().sum()),
        "retained_customers": int(data["Churn"].eq(0).sum()),
        "churned_customers": int(data["Churn"].eq(1).sum()),
        "churn_rate": float(data["Churn"].mean()),
    }
    return data, cleaning_summary


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(exclude=["object", "str", "category"]).columns.tolist()
    categorical_columns = features.select_dtypes(include=["str", "category"]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def calculate_metrics(
    model: Any,
    test_features: np.ndarray,
    test_target: pd.Series,
) -> tuple[dict[str, float | dict[str, Any]], np.ndarray]:
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]
    report = classification_report(
        test_target,
        predictions,
        target_names=["Not Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, float | dict[str, Any]] = {
        "accuracy": float(accuracy_score(test_target, predictions)),
        "precision": float(precision_score(test_target, predictions, zero_division=0)),
        "recall": float(recall_score(test_target, predictions, zero_division=0)),
        "f1_score": float(f1_score(test_target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_target, probabilities)),
        "classification_report": report,
    }
    return metrics, predictions


def save_churn_distribution(data: pd.DataFrame) -> None:
    counts = data["Churn"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Retained", "Churned"]
    values = counts.to_numpy()
    percentages = values / values.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=[COLORS["Retained"], COLORS["Churned"]], width=0.58)
    ax.set_title("Customer Churn Distribution", fontsize=15, weight="bold")
    ax.set_ylabel("Customers")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for bar, value, percentage in zip(bars, values, percentages, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:,}\n{percentage:.1%}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "churn_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_categorical_churn(data: pd.DataFrame, column: str, filename: str, title: str) -> None:
    summary = data.groupby(column, observed=True)["Churn"].agg(["size", "mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(summary[column], summary["mean"], color=COLORS["secondary"], width=0.62)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_ylabel("Churn rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, max(0.5, float(summary["mean"].max()) * 1.25))
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for bar, rate, customers in zip(bars, summary["mean"], summary["size"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.012,
            f"{rate:.1%}\n{int(customers):,} customers",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_distribution_by_churn(
    data: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
    xlabel: str,
) -> None:
    retained = data.loc[data["Churn"].eq(0), column]
    churned = data.loc[data["Churn"].eq(1), column]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    box = ax.boxplot(
        [retained, churned],
        tick_labels=["Retained", "Churned"],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, color in zip(box["boxes"], [COLORS["Retained"], COLORS["Churned"]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_ylabel(xlabel)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(model_results: dict[str, dict[str, Any]]) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    model_names = list(model_results)
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, model_name in enumerate(model_names):
        values = [float(model_results[model_name]["metrics"][name]) for name in metric_names]
        offset = (index - (len(model_names) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            values,
            width,
            label=model_name,
            color=COLORS["accent"] if index == 0 else COLORS["secondary"],
        )
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Model Performance Comparison", fontsize=15, weight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    test_target: pd.Series,
    probability_results: dict[str, np.ndarray],
    roc_results: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = [COLORS["accent"], COLORS["secondary"]]
    for color, (model_name, probabilities) in zip(colors, probability_results.items(), strict=True):
        false_positive_rate, true_positive_rate, _ = roc_curve(test_target, probabilities)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            linewidth=2.5,
            label=f"{model_name} (AUC = {roc_results[model_name]:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Random baseline")
    ax.set_title("ROC Curve", fontsize=15, weight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(
    test_target: pd.Series,
    probability_results: dict[str, np.ndarray],
    f1_results: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = [COLORS["accent"], COLORS["secondary"]]
    for color, (model_name, probabilities) in zip(colors, probability_results.items(), strict=True):
        precision, recall, _ = precision_recall_curve(test_target, probabilities)
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2.5,
            label=f"{model_name} (F1 = {f1_results[model_name]:.3f})",
        )
    ax.set_title("Precision-Recall Curve", fontsize=15, weight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.7)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "precision_recall_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(matrix: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(matrix, cmap="Blues")
    labels = ["Not Churn", "Churn"]
    ax.set_xticks(np.arange(2), labels)
    ax.set_yticks(np.arange(2), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix", fontsize=15, weight="bold")
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else COLORS["text"],
                fontsize=18,
                weight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(
    preprocessor: ColumnTransformer,
    random_forest: RandomForestClassifier,
) -> None:
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.removeprefix("num__").removeprefix("cat__") for name in feature_names
    ]
    importance = pd.DataFrame(
        {"Feature": clean_names, "Importance": random_forest.feature_importances_}
    )
    importance = importance.sort_values("Importance", ascending=False).head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(importance["Feature"], importance["Importance"], color=COLORS["accent"])
    ax.set_title("Random Forest Feature Importance", fontsize=15, weight="bold")
    ax.set_xlabel("Relative importance")
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in importance["Importance"]], padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_chart_style()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data, cleaning_summary = load_dataset(DATA_PATH)
    features = data.drop(columns=["Churn"])
    target = data["Churn"]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    preprocessor = build_preprocessor(features)
    processed_train = preprocessor.fit_transform(train_features)
    processed_test = preprocessor.transform(test_features)
    search = GridSearchCV(
        estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        param_grid={
            "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "class_weight": [None, "balanced"],
        },
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(processed_train, train_target)
    logistic_regression = search.best_estimator_
    random_forest = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(processed_train, train_target)
    models = {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }
    model_results: dict[str, dict[str, Any]] = {}
    probability_results: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        metrics, predictions = calculate_metrics(model, processed_test, test_target)
        model_results[model_name] = {
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(test_target, predictions).tolist(),
        }
        probability_results[model_name] = model.predict_proba(processed_test)[:, 1]
    selected_name = max(
        model_results,
        key=lambda name: float(model_results[name]["metrics"]["f1_score"]),
    )
    selected_model = models[selected_name]
    selected_metrics = model_results[selected_name]["metrics"]
    selected_matrix = np.asarray(model_results[selected_name]["confusion_matrix"])
    metadata = {
        "dataset": cleaning_summary,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "training_rows": len(train_features),
            "testing_rows": len(test_features),
        },
        "feature_columns": features.columns.tolist(),
        "models": model_results,
        "selected_model": selected_name,
        "selection_metric": "f1_score",
        "selected_model_score": selected_metrics["f1_score"],
        "best_logistic_regression_params": {
            key: str(value) if value is None else value
            for key, value in search.best_params_.items()
        },
        "confusion_matrix": selected_matrix.tolist(),
    }
    joblib.dump(selected_model, MODEL_DIR / "churn_model.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    save_churn_distribution(data)
    save_categorical_churn(data, "Contract", "churn_by_contract.png", "Churn Rate by Contract Type")
    save_categorical_churn(
        data,
        "InternetService",
        "churn_by_internet_service.png",
        "Churn Rate by Internet Service",
    )
    save_distribution_by_churn(
        data,
        "tenure",
        "tenure_vs_churn.png",
        "Customer Tenure by Churn Status",
        "Tenure (months)",
    )
    save_distribution_by_churn(
        data,
        "MonthlyCharges",
        "monthly_charges_vs_churn.png",
        "Monthly Charges by Churn Status",
        "Monthly charge",
    )
    save_feature_importance(preprocessor, random_forest)
    save_model_comparison(model_results)
    save_roc_curve(
        test_target,
        probability_results,
        {
            name: float(result["metrics"]["roc_auc"])
            for name, result in model_results.items()
        },
    )
    save_precision_recall_curve(
        test_target,
        probability_results,
        {
            name: float(result["metrics"]["f1_score"])
            for name, result in model_results.items()
        },
    )
    save_confusion_matrix(selected_matrix, selected_name)
    print(json.dumps(metadata, indent=2))
    print(f"Selected model: {selected_name}")
    print(f"Artifacts saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
