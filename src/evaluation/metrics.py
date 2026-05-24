from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str]
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        labels=list(range(len(label_names))),
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }


def compute_summary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """Compute lightweight metrics suitable for per-epoch training logs."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def print_classification_metrics(
    metrics: dict[str, Any],
    label_names: list[str] | None = None,
    include_report: bool = True
) -> None:
    """Print classification metrics in a compact notebook-friendly format."""
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    if not include_report or "classification_report" not in metrics:
        return

    report = metrics["classification_report"]
    if label_names is None:
        aggregate_rows = {"accuracy", "macro avg", "weighted avg"}
        label_names = [
            label_name
            for label_name, label_metrics in report.items()
            if (
                label_name not in aggregate_rows
                and isinstance(label_metrics, dict)
                and "precision" in label_metrics
            )
        ]

    print("\nClassification report:")
    rows = []
    for label_name in label_names:
        label_metrics = report[label_name]
        rows.append(
            {
                "emotion": label_name,
                "precision": label_metrics["precision"],
                "recall": label_metrics["recall"],
                "f1_score": label_metrics["f1-score"],
                "support": int(label_metrics["support"])
            }
        )

    table = pd.DataFrame(rows)
    print(
        table.to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format
            }
        )
    )


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def save_classification_report_csv(metrics: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    report.to_csv(output_path)
    return output_path


def save_confusion_matrix_plot(
    metrics: dict[str, Any],
    label_names: list[str],
    output_path: str | Path,
    title: str = "Confusion matrix",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax
    )
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_classification_evaluation_outputs(
    metrics: dict[str, Any],
    split_metadata: pd.DataFrame,
    y_pred: np.ndarray,
    prediction_scores: np.ndarray,
    label_names: list[str],
    output_dir: str | Path,
    split: str,
    model_name: str,
    score_prefix: str
) -> dict[str, Path]:
    """Save standard classification evaluation artifacts for one split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_metrics(metrics, output_dir / f"{split}_metrics.json")
    report_path = save_classification_report_csv(
        metrics,
        output_dir / f"{split}_classification_report.csv"
    )
    confusion_matrix_path = save_confusion_matrix_plot(
        metrics,
        label_names,
        output_dir / f"{split}_confusion_matrix.png",
        title=f"{model_name} {split} confusion matrix"
    )

    predictions = split_metadata[["file_name", "emotion", "label"]].copy()
    predictions["predicted_label"] = y_pred
    predictions["predicted_emotion"] = [label_names[index] for index in y_pred]
    for index, label_name in enumerate(label_names):
        predictions[f"{score_prefix}_{label_name}"] = prediction_scores[:, index]

    predictions_path = output_dir / f"{split}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    return {
        "metrics": metrics_path,
        "classification_report": report_path,
        "confusion_matrix": confusion_matrix_path,
        "predictions": predictions_path
    }
