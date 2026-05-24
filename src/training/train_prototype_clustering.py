from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.data.crema_d import EMOTION_NAMES, load_features
from src.evaluation.metrics import compute_summary_classification_metrics
from src.models.prototype_clustering import (
    PrototypeClusteringClassifier,
    PrototypeClusteringMetadata,
    l2_normalize_rows,
)


@dataclass
class PrototypeClusteringTrainingConfig:
    embedding_dim: int = 128
    cluster_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 8, 10)
    top_ns: tuple[int, ...] = (1, 3, 5, 7, 9)
    num_classes: int = 6
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300
    monitor_metric: str = "macro_f1"
    verbose: bool = True


def _validate_metadata(metadata: pd.DataFrame) -> None:
    required_columns = {"file_name", "label", "emotion", "split"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Embedding metadata is missing columns: {sorted(missing_columns)}")


def _build_centroids(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    config: PrototypeClusteringTrainingConfig
) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    centroid_labels = []

    for label in range(config.num_classes):
        class_embeddings = train_embeddings[train_labels == label]
        if len(class_embeddings) < k:
            raise ValueError(
                f"Cannot fit {k} clusters for class {label}: "
                f"only {len(class_embeddings)} training samples available"
            )

        kmeans = KMeans(
            n_clusters=k,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state
        )
        kmeans.fit(class_embeddings)
        centroids.append(kmeans.cluster_centers_.astype(np.float32))
        centroid_labels.extend([label] * k)

    return (
        l2_normalize_rows(np.vstack(centroids)),
        np.asarray(centroid_labels, dtype=np.int64)
    )


def _build_centroid_classifier(
    centroids: np.ndarray,
    centroid_labels: np.ndarray,
    top_n: int,
    config: PrototypeClusteringTrainingConfig
) -> PrototypeClusteringClassifier:
    metadata = PrototypeClusteringMetadata(
        top_n=top_n,
        label_names=EMOTION_NAMES,
        embedding_dim=config.embedding_dim,
    )
    return PrototypeClusteringClassifier(
        centroids=centroids,
        centroid_labels=centroid_labels,
        metadata=metadata
    )


def train_prototype_clustering(
    embedding_dir: str | Path,
    output_dir: str | Path,
    config: PrototypeClusteringTrainingConfig | None = None
) -> dict[str, Any]:
    """Run validation grid search for the prototype clustering classifier."""
    config = config or PrototypeClusteringTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_features(embedding_dir)
    _validate_metadata(metadata)
    if embeddings.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Expected embedding dim {config.embedding_dim}, got {embeddings.shape[1]}"
        )

    train_indices = metadata.index[metadata["split"] == "train"].to_numpy()
    val_indices = metadata.index[metadata["split"] == "val"].to_numpy()
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise ValueError("Both train and val splits must contain samples")

    train_embeddings = embeddings[train_indices]
    train_labels = metadata.loc[train_indices, "label"].to_numpy(dtype=np.int64)
    val_embeddings = embeddings[val_indices]
    val_labels = metadata.loc[val_indices, "label"].to_numpy(dtype=np.int64)

    results = []
    best_score = -np.inf
    best_row: dict[str, Any] | None = None
    best_classifier: PrototypeClusteringClassifier | None = None

    for k in config.cluster_counts:
        if k <= 0:
            raise ValueError(f"cluster_counts must be positive, got {k}")

        centroids, centroid_labels = _build_centroids(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            k=k,
            config=config
        )
        for top_n in config.top_ns:
            if top_n <= 0:
                raise ValueError(f"top_ns must be positive, got {top_n}")
            if top_n > k * config.num_classes:
                continue

            classifier = _build_centroid_classifier(
                centroids=centroids,
                centroid_labels=centroid_labels,
                top_n=top_n,
                config=config
            )

            predictions = classifier.predict(val_embeddings)
            metrics = compute_summary_classification_metrics(val_labels, predictions)
            row = {
                "k": k,
                "top_n": top_n,
                "val_accuracy": metrics["accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_weighted_f1": metrics["weighted_f1"],
                "num_centroids": k * config.num_classes,
            }
            results.append(row)

            score_key = f"val_{config.monitor_metric}"
            if score_key not in row:
                raise ValueError(
                    f"Unsupported monitor_metric '{config.monitor_metric}'. "
                    "Use one of: accuracy, macro_f1, weighted_f1"
                )
            score = float(row[score_key])
            tie_breaker = float(row["val_accuracy"])
            current_best_accuracy = -np.inf if best_row is None else float(best_row["val_accuracy"])
            improved = score > best_score or (
                np.isclose(score, best_score) and tie_breaker > current_best_accuracy
            )
            if improved:
                best_score = score
                best_row = row
                best_classifier = classifier

            if config.verbose:
                print(
                    f"K={k:02d}, top-N={top_n:02d} | "
                    f"val acc {metrics['accuracy']:.4f}, "
                    f"macro F1 {metrics['macro_f1']:.4f}, "
                    f"weighted F1 {metrics['weighted_f1']:.4f}"
                )

    if best_row is None or best_classifier is None:
        raise RuntimeError("Grid search finished without a valid configuration")

    search_results_path = output_dir / "grid_search_results.csv"
    training_config_path = output_dir / "training_config.json"
    pd.DataFrame(results).sort_values(
        by=[f"val_{config.monitor_metric}", "val_accuracy"],
        ascending=False
    ).to_csv(search_results_path, index=False)
    training_config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    best_classifier.save(
        output_dir,
        extra_config={
            "best_validation": best_row,
            "embedding_dir": str(embedding_dir),
            "grid_search_results": str(search_results_path),
        }
    )

    if config.verbose:
        print(
            "\nBest prototype clustering configuration\n"
            f"  K:           {best_row['k']}\n"
            f"  Top-N:       {best_row['top_n']}\n"
            f"  Validation:  accuracy {best_row['val_accuracy']:.4f}, "
            f"macro F1 {best_row['val_macro_f1']:.4f}, "
            f"weighted F1 {best_row['val_weighted_f1']:.4f}"
        )

    return {
        "model_dir": output_dir,
        "grid_search_results": search_results_path,
        "training_config": training_config_path,
        "best_config": best_row,
    }
