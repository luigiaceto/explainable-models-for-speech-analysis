from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np


def l2_normalize_rows(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, eps, None)


@dataclass(frozen=True)
class PrototypeClusteringMetadata:
    top_n: int
    label_names: list[str]
    embedding_dim: int
    similarity: str = "cosine"
    normalization: str = "l2"
    score_rule: str = "top_n_similarity_sum"


class PrototypeClusteringClassifier:
    """Classifier based on cosine similarities to emotion-specific prototypes."""

    def __init__(
        self,
        metadata: PrototypeClusteringMetadata,
        prototypes: np.ndarray,
        prototype_labels: np.ndarray
    ) -> None:
        self.prototypes = l2_normalize_rows(prototypes)
        self.prototype_labels = np.asarray(prototype_labels, dtype=np.int64)
        self.metadata = metadata

        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must be a 2D array")
        if len(self.prototypes) != len(self.prototype_labels):
            raise ValueError("prototypes and prototype_labels have different lengths")
        if self.prototypes.shape[1] != self.metadata.embedding_dim:
            raise ValueError(
                f"Expected prototype dim {self.metadata.embedding_dim}, "
                f"got {self.prototypes.shape[1]}"
            )

    @property
    def num_classes(self) -> int:
        return len(self.metadata.label_names)

    def similarities(self, embeddings: np.ndarray) -> np.ndarray:
        normalized_embeddings = l2_normalize_rows(embeddings)
        return normalized_embeddings @ self.prototypes.T

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        similarities = self.similarities(embeddings)
        top_indices = np.argpartition(
            -similarities,
            kth=self.metadata.top_n - 1,
            axis=1
        )[:, : self.metadata.top_n]

        scores = np.zeros((len(similarities), self.num_classes), dtype=np.float32)
        row_indices = np.arange(len(similarities))
        for rank in range(self.metadata.top_n):
            prototype_indices = top_indices[:, rank]
            labels = self.prototype_labels[prototype_indices]
            values = similarities[row_indices, prototype_indices]
            np.add.at(scores, (row_indices, labels), values)
        return scores

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return self.scores(embeddings).argmax(axis=1) # return predicted label for each sample passed to the function

    def save(self, output_dir: str | Path, extra_config: dict[str, Any] | None = None) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prototypes_path = output_dir / "prototypes.npy"
        prototype_labels_path = output_dir / "prototype_labels.npy"
        config_path = output_dir / "prototype_config.json"

        np.save(prototypes_path, self.prototypes.astype(np.float32))
        np.save(prototype_labels_path, self.prototype_labels.astype(np.int64))

        config = {
            "metadata": asdict(self.metadata),
            "num_prototypes": int(len(self.prototypes)),
            "classification_vectors": "prototypes.npy",
        }
        if extra_config is not None:
            config["extra"] = extra_config
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        return {
            "prototypes": prototypes_path,
            "prototype_labels": prototype_labels_path,
            "config": config_path
        }


def load_prototype_clustering_classifier(
    model_dir: str | Path
) -> tuple[PrototypeClusteringClassifier, dict[str, Any]]:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "prototype_config.json").read_text(encoding="utf-8"))
    metadata = PrototypeClusteringMetadata(**config["metadata"])
    classifier = PrototypeClusteringClassifier(
        metadata=metadata,
        prototypes=np.load(model_dir / "prototypes.npy"),
        prototype_labels=np.load(model_dir / "prototype_labels.npy")
    )
    return classifier, config
