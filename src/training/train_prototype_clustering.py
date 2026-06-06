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
    l2_normalize_rows
)


@dataclass
class PrototypeClusteringTrainingConfig:
    embedding_dim: int = 128
    cluster_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 8, 10)
    num_classes: int = len(EMOTION_NAMES)
    random_state: int = 42
    n_init: int = 10 # parameter of KMeans, try #n_init different initializations and keep the best
    max_iter: int = 300 # since KMeans stops early when convergence is reached, this value does not imply that all runs perform 300 iterations
    monitor_metric: str = "macro_f1" # accuracy, macro_f1
    verbose: bool = True


def _build_centroids(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    config: PrototypeClusteringTrainingConfig
) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    centroid_labels = []

    # for each emotion, run KMeans in order to obtain k centroids
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
        kmeans.fit(class_embeddings) # runs the clustering
        centroids.append(kmeans.cluster_centers_.astype(np.float32))
        centroid_labels.extend([label] * k) # add centroid labels

    return (
        l2_normalize_rows(np.vstack(centroids)), # centroids produced by KMeans aren't normalized, even if the embeddings are l2
        np.asarray(centroid_labels, dtype=np.int64)
    )


def _build_prototype_classifier(
    prototypes: np.ndarray, # not centroids, but medoids
    prototype_labels: np.ndarray,
    config: PrototypeClusteringTrainingConfig
) -> PrototypeClusteringClassifier:
    metadata = PrototypeClusteringMetadata(
        label_names=EMOTION_NAMES,
        embedding_dim=config.embedding_dim
    )
    return PrototypeClusteringClassifier(
        metadata=metadata,
        prototypes=prototypes,
        prototype_labels=prototype_labels
    )


def _map_centroids_to_real_prototypes(
    centroids: np.ndarray,
    centroid_labels: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_metadata: pd.DataFrame,
    config: PrototypeClusteringTrainingConfig
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Map each class-specific centroid to a real training sample of the same class."""
    normalized_centroids = l2_normalize_rows(centroids)
    normalized_train_embeddings = l2_normalize_rows(train_embeddings)
    prototypes = np.zeros_like(normalized_centroids, dtype=np.float32)
    prototype_labels = np.zeros_like(centroid_labels, dtype=np.int64)
    records = []

    # for each emotion
    for label in range(config.num_classes):
        centroid_positions = np.where(centroid_labels == label)[0]
        class_positions = np.where(train_labels == label)[0]
        class_embeddings = normalized_train_embeddings[class_positions]
        similarities = normalized_centroids[centroid_positions] @ class_embeddings.T # cosine similarity
        used_local_positions: set[int] = set() # samples already used as prototypes

        # Assign the most confident centroid-sample pairs first, while keeping
        # prototypes unique within each emotion class.
        # Create a list of (centroid, samples) from the most similar to the least similar.
        pair_order = np.dstack(
            np.unravel_index(
                np.argsort(-similarities, axis=None),
                similarities.shape
            )
        )[0]
        assigned_centroids: set[int] = set() # centroids already assigned to samples

        for centroid_local_position, sample_local_position in pair_order:
            centroid_local_position = int(centroid_local_position)
            sample_local_position = int(sample_local_position)
            if centroid_local_position in assigned_centroids:
                continue
            if sample_local_position in used_local_positions:
                continue

            global_centroid_position = int(centroid_positions[centroid_local_position])
            global_sample_position = int(class_positions[sample_local_position])
            sample_row = train_metadata.iloc[global_sample_position]

            prototypes[global_centroid_position] = normalized_train_embeddings[global_sample_position]
            prototype_labels[global_centroid_position] = label
            used_local_positions.add(sample_local_position)
            assigned_centroids.add(centroid_local_position)
            records.append(
                {
                    "prototype_position": global_centroid_position,
                    "centroid_label": label,
                    "prototype_label": label,
                    "file_name": sample_row["file_name"],
                    "emotion": sample_row["emotion"],
                    "train_embedding_position": global_sample_position,
                    "dataset_metadata_index": int(sample_row["metadata_index"]),
                    "centroid_similarity": float(
                        similarities[centroid_local_position, sample_local_position]
                    )
                }
            )

            if len(assigned_centroids) == len(centroid_positions):
                break

        if len(assigned_centroids) != len(centroid_positions):
            raise RuntimeError(f"Could not assign all prototypes for class {label}")

    prototype_metadata = (
        pd.DataFrame(records)
        .sort_values("prototype_position")
        .reset_index(drop=True)
    )
    return prototypes, prototype_labels, prototype_metadata


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
    if embeddings.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Expected embedding dim {config.embedding_dim}, got {embeddings.shape[1]}"
        )

    train_indices = metadata.index[metadata["split"] == "train"].to_numpy()
    val_indices = metadata.index[metadata["split"] == "val"].to_numpy()

    train_embeddings = embeddings[train_indices]
    train_labels = metadata.loc[train_indices, "label"].to_numpy(dtype=np.int64)
    train_metadata = metadata.loc[train_indices].reset_index(drop=False).rename(
        columns={"index": "metadata_index"}
    )
    val_embeddings = embeddings[val_indices]
    val_labels = metadata.loc[val_indices, "label"].to_numpy(dtype=np.int64)

    results = []
    best_score = -np.inf
    best_row: dict[str, Any] | None = None
    best_classifier: PrototypeClusteringClassifier | None = None
    best_prototype_metadata: pd.DataFrame | None = None
    best_centroids: np.ndarray | None = None
    best_centroid_labels: np.ndarray | None = None

    for k in config.cluster_counts:
        if k <= 0:
            raise ValueError(f"cluster_counts must be positive, got {k}")

        centroids, centroid_labels = _build_centroids(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            k=k,
            config=config
        )
        prototypes, prototype_labels, prototype_metadata = _map_centroids_to_real_prototypes(
            centroids=centroids,
            centroid_labels=centroid_labels,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            train_metadata=train_metadata,
            config=config
        )
        num_prototypes = k * config.num_classes
        classifier = _build_prototype_classifier(
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            config=config
        )

        predictions = classifier.predict(val_embeddings)
        metrics = compute_summary_classification_metrics(val_labels, predictions)
        row = {
            "k": k,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "num_centroids": num_prototypes
        }
        results.append(row)

        score_key = f"val_{config.monitor_metric}"
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
            best_prototype_metadata = prototype_metadata
            best_centroids = centroids
            best_centroid_labels = centroid_labels

        if config.verbose:
            print(
                f"K={k:02d}, all prototypes={num_prototypes:02d} | "
                f"val acc {metrics['accuracy']:.4f}, "
                f"macro F1 {metrics['macro_f1']:.4f}"
            )

    if (
        best_row is None
        or best_classifier is None
        or best_prototype_metadata is None
        or best_centroids is None
        or best_centroid_labels is None
    ):
        raise RuntimeError("Grid search finished without a valid configuration")

    search_results_path = output_dir / "grid_search_results.csv"
    training_config_path = output_dir / "training_config.json"
    prototype_metadata_path = output_dir / "prototype_metadata.csv"
    centroids_path = output_dir / "centroids.npy"
    centroid_labels_path = output_dir / "centroid_labels.npy"
    pd.DataFrame(results).sort_values(
        by=[f"val_{config.monitor_metric}", "val_accuracy"],
        ascending=False
    ).to_csv(search_results_path, index=False)
    training_config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    best_prototype_metadata.to_csv(prototype_metadata_path, index=False)
    np.save(centroids_path, best_centroids.astype(np.float32))
    np.save(centroid_labels_path, best_centroid_labels.astype(np.int64))

    best_classifier.save(
        output_dir,
        extra_config={
            "best_validation": best_row,
            "embedding_dir": str(embedding_dir),
            "grid_search_results": str(search_results_path),
            "prototype_metadata": str(prototype_metadata_path),
            "cluster_centroids": str(centroids_path),
            "cluster_centroid_labels": str(centroid_labels_path),
            "prototype_source": "nearest_train_sample_to_class_kmeans_centroid"
        }
    )

    if config.verbose:
        print(
            "\nBest prototype clustering configuration\n"
            f"  K:           {best_row['k']}\n"
            f"  Prototypes:  {best_row['num_centroids']}\n"
            f"  Validation:  accuracy {best_row['val_accuracy']:.4f}, "
            f"macro F1 {best_row['val_macro_f1']:.4f}"
        )

    return {
        "model_dir": output_dir,
        "grid_search_results": search_results_path,
        "training_config": training_config_path,
        "prototype_metadata": prototype_metadata_path,
        "centroids": centroids_path,
        "centroid_labels": centroid_labels_path,
        "best_config": best_row
    }
