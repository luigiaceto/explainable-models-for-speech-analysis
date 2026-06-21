from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from src.data.crema_d import EMOTION_NAMES, load_features
from src.evaluation.evaluate_blackbox import load_blackbox_model
from src.models.prototype_clustering import load_prototype_clustering_classifier


def _load_aligned_split_metadata(
    feature_metadata: pd.DataFrame,
    splits_csv: str | Path
) -> pd.DataFrame:
    split_metadata = pd.read_csv(splits_csv)

    if len(split_metadata) != len(feature_metadata):
        raise ValueError("Split metadata and feature metadata have different lengths")
    if feature_metadata["file_name"].tolist() != split_metadata["file_name"].tolist():
        raise ValueError(
            f"Split metadata in {splits_csv} does not match the feature metadata"
        )
    if "split" not in split_metadata.columns:
        raise ValueError(f"Split metadata in {splits_csv} is missing the 'split' column")

    return split_metadata


def _validate_embedding_alignment(
    split_metadata: pd.DataFrame,
    embedding_metadata: pd.DataFrame,
    embedding_dir: str | Path
) -> None:
    if len(split_metadata) != len(embedding_metadata):
        raise ValueError(
            "Split metadata and prototype embedding metadata have different lengths"
        )
    if split_metadata["file_name"].tolist() != embedding_metadata["file_name"].tolist():
        raise ValueError(
            f"Prototype embedding metadata in {embedding_dir} is not aligned with the split metadata"
        )
    if "split" in embedding_metadata.columns:
        if split_metadata["split"].tolist() != embedding_metadata["split"].tolist():
            raise ValueError(
                f"Prototype embedding split labels in {embedding_dir} do not match the split metadata"
            )


def _predict_blackbox(
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    model: torch.nn.Module,
    device: torch.device
) -> np.ndarray:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_features = torch.as_tensor(
                np.asarray(features[batch_indices], dtype=np.float32),
                dtype=torch.float32,
                device=device
            )
            logits = model(batch_features)
            predictions.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(predictions).astype(np.int64)


def evaluate_clustering_surrogate_fidelity(
    blackbox_checkpoint_path: str | Path,
    feature_dir: str | Path,
    prototype_model_dir: str | Path,
    embedding_dir: str | Path,
    splits_csv: str | Path | None = None,
    split: str = "test",
    batch_size: int = 256,
    device: str | None = None
) -> dict[str, Any]:
    """Measure how often the prototype surrogate matches black-box predictions.

    The black-box prediction is treated as the target label. This evaluates
    global fidelity of the clustering surrogate on one split, usually ``test``.
    """
    blackbox_checkpoint_path = Path(blackbox_checkpoint_path)
    model, checkpoint, compute_device = load_blackbox_model(
        blackbox_checkpoint_path,
        device
    )
    features, feature_metadata = load_features(feature_dir, mmap_mode="r")

    if splits_csv is None:
        splits_csv = checkpoint.get("splits_path")
    if splits_csv is None:
        splits_csv = blackbox_checkpoint_path.parent / "splits.csv"

    split_metadata = _load_aligned_split_metadata(feature_metadata, splits_csv)
    if split not in set(split_metadata["split"]):
        raise ValueError(f"Split '{split}' not found in {splits_csv}")

    embeddings, embedding_metadata = load_features(embedding_dir, mmap_mode="r")
    _validate_embedding_alignment(split_metadata, embedding_metadata, embedding_dir)

    indices = split_metadata.index[split_metadata["split"] == split].to_numpy()
    if len(indices) == 0:
        raise ValueError(f"Split '{split}' in {splits_csv} does not contain any samples")

    blackbox_predictions = _predict_blackbox(
        features=features,
        indices=indices,
        batch_size=batch_size,
        model=model,
        device=compute_device
    )
    classifier, _ = load_prototype_clustering_classifier(prototype_model_dir)
    surrogate_predictions = classifier.predict(embeddings[indices])

    agreements = surrogate_predictions == blackbox_predictions
    num_correct = int(agreements.sum())
    num_samples = int(len(agreements))
    fidelity_accuracy = float(num_correct / num_samples)

    return {
        "split": split,
        "accuracy": fidelity_accuracy,
        "num_correct": num_correct,
        "num_samples": num_samples,
        "blackbox_predictions": blackbox_predictions.tolist(),
        "surrogate_predictions": surrogate_predictions.tolist(),
        "blackbox_emotions": [
            EMOTION_NAMES[index] for index in blackbox_predictions
        ],
        "surrogate_emotions": [
            EMOTION_NAMES[index] for index in surrogate_predictions
        ]
    }


def print_clustering_surrogate_fidelity_accuracy(
    blackbox_checkpoint_path: str | Path,
    feature_dir: str | Path,
    prototype_model_dir: str | Path,
    embedding_dir: str | Path,
    splits_csv: str | Path | None = None,
    split: str = "test",
    batch_size: int = 256,
    device: str | None = None
) -> dict[str, Any]:
    """Print the prototype surrogate accuracy against black-box predictions."""
    metrics = evaluate_clustering_surrogate_fidelity(
        blackbox_checkpoint_path=blackbox_checkpoint_path,
        feature_dir=feature_dir,
        prototype_model_dir=prototype_model_dir,
        embedding_dir=embedding_dir,
        splits_csv=splits_csv,
        split=split,
        batch_size=batch_size,
        device=device
    )
    
    print(
        "Accuracy against black-box predictions: "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['num_correct']}/{metrics['num_samples']})"
    )
    return metrics
