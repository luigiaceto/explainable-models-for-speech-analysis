from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from src.data.tess import EMOTION_NAMES, load_features, make_tess_feature_loader
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_classification_evaluation_outputs
)
from src.models.blackbox import BlackBoxEmotionClassifier
from src.utils.utils import device_or_default


def load_blackbox_model(
    checkpoint_path: str | Path,
    device: str | None = None
) -> tuple[BlackBoxEmotionClassifier, dict[str, Any], torch.device]:
    compute_device = device_or_default(device)
    checkpoint = torch.load(checkpoint_path, map_location=compute_device)
    config = checkpoint["config"]
    model = BlackBoxEmotionClassifier(
        input_dim=config["input_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        activation=config["activation"]
    ).to(compute_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, compute_device


def evaluate_blackbox(
    checkpoint_path: str | Path,
    feature_dir: str | Path,
    splits_csv: str | Path | None = None,
    split: str = "test",
    output_dir: str | Path | None = None,
    batch_size: int = 256,
    device: str | None = None
) -> dict[str, Any]:
    model, checkpoint, compute_device = load_blackbox_model(checkpoint_path, device)
    features, feature_metadata = load_features(feature_dir)

    if splits_csv is None:
        splits_csv = checkpoint.get("splits_path")
    if splits_csv is None:
        splits_csv = Path(checkpoint_path).parent / "splits.csv"

    # performing checks in order to prevent:
    # 1. using the wrong splits.csv
    # 2. using unaligned features and metadata
    # 3. evaluating an empty or non-existent split
    split_metadata = pd.read_csv(splits_csv)
    if len(split_metadata) != len(feature_metadata):
        raise ValueError("Split metadata and feature metadata have different lengths")
    if feature_metadata["file_name"].tolist() != split_metadata["file_name"].tolist():
        raise ValueError(
            f"Split metadata in {splits_csv} does not match feature metadata in {feature_dir}"
        )
    if "split" not in split_metadata.columns:
        raise ValueError(f"Split metadata in {splits_csv} is missing the 'split' column")
    if split not in set(split_metadata["split"]):
        raise ValueError(f"Split '{split}' not found in {splits_csv}")

    indices = split_metadata.index[split_metadata["split"] == split].tolist()
    if not indices:
        raise ValueError(f"Split '{split}' in {splits_csv} does not contain any samples")

    loader = make_tess_feature_loader(
        features=features,
        metadata=split_metadata,
        split_name=split,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    y_true = []
    y_pred = []
    probabilities = []
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(compute_device)
            logits = model(batch_features)
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(batch_probabilities)
            y_pred.append(batch_probabilities.argmax(axis=1))
            y_true.append(batch_labels.numpy())

    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    probability_array = np.concatenate(probabilities)
    metrics = compute_classification_metrics(y_true_array, y_pred_array, EMOTION_NAMES)

    if output_dir is not None:
        save_classification_evaluation_outputs(
            metrics,
            split_metadata.loc[indices],
            y_pred_array,
            probability_array,
            label_names=EMOTION_NAMES,
            output_dir=output_dir,
            split=split,
            model_name="Black-box",
            score_prefix="probability"
        )

    return metrics
