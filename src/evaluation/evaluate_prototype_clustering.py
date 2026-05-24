from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.data.crema_d import EMOTION_NAMES, load_features
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_classification_report_csv,
    save_confusion_matrix_plot,
    save_metrics,
)
from src.models.prototype_clustering import load_prototype_clustering_classifier


def print_prototype_clustering_metrics(metrics: dict[str, Any]) -> None:
    """Print prototype clustering metrics in a compact format."""
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


def evaluate_prototype_clustering(
    model_dir: str | Path,
    embedding_dir: str | Path,
    split: str = "test",
    output_dir: str | Path | None = None
) -> dict[str, Any]:
    classifier, _ = load_prototype_clustering_classifier(model_dir)
    embeddings, metadata = load_features(embedding_dir)

    if "split" not in metadata.columns:
        raise ValueError(f"Embedding metadata in {embedding_dir} is missing the 'split' column")
    if split not in set(metadata["split"]):
        raise ValueError(f"Split '{split}' not found in {embedding_dir}")

    indices = metadata.index[metadata["split"] == split].to_numpy()
    if len(indices) == 0:
        raise ValueError(f"Split '{split}' does not contain any samples")

    split_embeddings = embeddings[indices]
    split_metadata = metadata.iloc[indices].reset_index(drop=True)
    y_true = split_metadata["label"].to_numpy(dtype=np.int64)
    scores = classifier.scores(split_embeddings)
    y_pred = scores.argmax(axis=1)
    metrics = compute_classification_metrics(y_true, y_pred, EMOTION_NAMES)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, output_dir / f"{split}_metrics.json")
        save_classification_report_csv(metrics, output_dir / f"{split}_classification_report.csv")
        save_confusion_matrix_plot(
            metrics,
            EMOTION_NAMES,
            output_dir / f"{split}_confusion_matrix.png",
            title=f"Prototype clustering {split} confusion matrix"
        )

        predictions = split_metadata[["file_name", "emotion", "label"]].copy()
        predictions["predicted_label"] = y_pred
        predictions["predicted_emotion"] = [EMOTION_NAMES[index] for index in y_pred]
        for index, emotion in enumerate(EMOTION_NAMES):
            predictions[f"score_{emotion}"] = scores[:, index]
        predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False)

    return metrics
