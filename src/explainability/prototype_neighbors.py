from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.data.crema_d import EMOTION_NAMES
from src.data.common import load_features
from src.models.prototype_clustering import load_prototype_clustering_classifier


def _find_sample_index(metadata: pd.DataFrame, file_name: str) -> int:
    matches = metadata.index[metadata["file_name"] == file_name].to_numpy()
    if len(matches) == 0:
        raise ValueError(f"Sample not found: {file_name}")
    if len(matches) > 1:
        raise ValueError(f"Multiple samples found for file name: {file_name}")
    return int(matches[0])


def _load_prototype_metadata(model_dir: Path) -> pd.DataFrame | None:
    metadata_path = model_dir / "prototype_metadata.csv"
    if not metadata_path.exists():
        return None
    return pd.read_csv(metadata_path)


def explain_sample_by_filename(
    embedding_metadata: pd.DataFrame,
    sample_to_explain: str,
    model_dir: str | Path,
    embedding_dir: str | Path
) -> dict[str, Any]:
    """Return labels, class scores, and top-N prototype neighbors for one sample.

    This function reuses saved embeddings and prototype files. It does not load
    the original WAV file, the audio encoder, or the black-box classifier.
    """

    if sample_to_explain is None:
        file_name = embedding_metadata.loc[
            embedding_metadata["split"] == "test",
            "file_name"
        ].iloc[0]
    else:
        if sample_to_explain not in set(embedding_metadata["file_name"]):
            raise ValueError(f"Sample not found in saved embeddings: {sample_to_explain}")
        file_name = sample_to_explain

    model_dir = Path(model_dir)
    classifier, _ = load_prototype_clustering_classifier(model_dir)
    embeddings, metadata = load_features(embedding_dir)
    prototype_metadata = _load_prototype_metadata(model_dir)

    sample_index = _find_sample_index(metadata, file_name)
    sample_embedding = embeddings[sample_index : sample_index + 1]
    similarities = classifier.similarities(sample_embedding)[0]
    scores = classifier.scores(sample_embedding)[0]
    predicted_label = int(np.argmax(scores))
    true_label = int(metadata.loc[sample_index, "label"])

    top_n = classifier.metadata.top_n
    top_indices = np.argpartition(-similarities, kth=top_n - 1)[:top_n]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    prototype_metadata_by_position = prototype_metadata.set_index("prototype_position")

    top_prototypes = []
    for rank, prototype_index in enumerate(top_indices, start=1):
        prototype_index = int(prototype_index)
        prototype_label = int(classifier.prototype_labels[prototype_index])

        if prototype_index not in prototype_metadata_by_position.index:
            raise ValueError(f"Missing metadata for prototype {prototype_index}")

        row = prototype_metadata_by_position.loc[prototype_index]

        prototype_info: dict[str, Any] = {
            "rank": rank,
            "prototype_position": prototype_index,
            "prototype_label": prototype_label,
            "prototype_emotion": EMOTION_NAMES[prototype_label],
            "similarity": float(similarities[prototype_index]),
            "prototype_file_name": row["file_name"],
            "prototype_source_emotion": row["emotion"],
            "centroid_similarity": float(row["centroid_similarity"])
        }

        top_prototypes.append(prototype_info)

    return {
        "file_name": file_name,
        "true_label": true_label,
        "true_emotion": EMOTION_NAMES[true_label],
        "predicted_label": predicted_label,
        "predicted_emotion": EMOTION_NAMES[predicted_label],
        "top_n": top_n,
        "scores": {
            emotion: float(scores[index])
            for index, emotion in enumerate(EMOTION_NAMES)
        },
        "top_prototypes": top_prototypes
    }


def print_prototype_explanation(explanation: dict[str, Any]) -> None:
    """Print a compact, notebook-friendly prototype explanation."""
    print(f"Sample:    {explanation['file_name']}")
    print(f"True:      {explanation['true_emotion']}")
    print(f"Predicted: {explanation['predicted_emotion']}")
    print("\nClass scores:")
    for emotion, score in explanation["scores"].items():
        print(f"  {emotion:>7}: {score:.4f}")

    print(f"\nTop-{explanation['top_n']} prototypes:")
    for prototype in explanation["top_prototypes"]:
        file_name = prototype.get("prototype_file_name", "<metadata unavailable>")
        print(
            f"  #{prototype['rank']:02d} "
            f"{prototype['prototype_emotion']:>7} | "
            f"sim {prototype['similarity']:.4f} | "
            f"{file_name}"
        )
