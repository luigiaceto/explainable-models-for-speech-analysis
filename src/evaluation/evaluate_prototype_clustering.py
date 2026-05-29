from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from src.data.crema_d import EMOTION_NAMES
from src.data.common import load_features
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_classification_evaluation_outputs,
)
from src.models.prototype_clustering import load_prototype_clustering_classifier


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
        save_classification_evaluation_outputs(
            metrics,
            split_metadata,
            y_pred,
            scores,
            label_names=EMOTION_NAMES,
            output_dir=output_dir,
            split=split,
            model_name="Prototype clustering",
            score_prefix="score"
        )

    return metrics
