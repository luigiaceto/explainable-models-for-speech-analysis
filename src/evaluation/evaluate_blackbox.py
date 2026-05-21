from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.crema_d import CremaDFeatureDataset, EMOTION_NAMES, load_features
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_classification_report_csv,
    save_confusion_matrix_plot,
    save_metrics
)
from src.models.blackbox import BlackBoxEmotionClassifier
from src.utils.utils import device_or_default


def print_classification_metrics(metrics: dict[str, Any]) -> None:
    """Print classification metrics in a compact tabular format."""
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nClassification report:")

    report = metrics["classification_report"]
    rows = []
    for label_name in EMOTION_NAMES:
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

    split_metadata = pd.read_csv(splits_csv)
    if len(split_metadata) != len(feature_metadata):
        raise ValueError("Split metadata and feature metadata have different lengths")
    if feature_metadata["file_name"].tolist() != split_metadata["file_name"].tolist():
        raise ValueError(f"Split '{split}' not found in {splits_csv}")

    indices = split_metadata.index[split_metadata["split"] == split].tolist()
    dataset = CremaDFeatureDataset(features, split_metadata, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, output_dir / f"{split}_metrics.json")
        save_classification_report_csv(metrics, output_dir / f"{split}_classification_report.csv")
        save_confusion_matrix_plot(
            metrics,
            EMOTION_NAMES,
            output_dir / f"{split}_confusion_matrix.png",
            title=f"Black-box {split} confusion matrix"
        )
        predictions = split_metadata.loc[indices, ["file_name", "emotion", "label"]].copy()
        predictions["predicted_label"] = y_pred_array
        predictions["predicted_emotion"] = [EMOTION_NAMES[index] for index in y_pred_array]
        for index, emotion in enumerate(EMOTION_NAMES):
            predictions[f"probability_{emotion}"] = probability_array[:, index]
        predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False)

    return metrics
