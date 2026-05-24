from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from src.data.crema_d import EMOTION_NAMES, load_features
from src.models.prototype_clustering import load_prototype_clustering_classifier


def score_sample_by_filename(
    file_name: str,
    model_dir: str | Path,
    embedding_dir: str | Path
) -> dict[str, Any]:
    """Return class scores and labels for one sample identified by file name."""
    classifier, _ = load_prototype_clustering_classifier(model_dir)
    embeddings, metadata = load_features(embedding_dir)

    matches = metadata.index[metadata["file_name"] == file_name].to_numpy()
    if len(matches) == 0:
        raise ValueError(f"Sample not found: {file_name}")
    if len(matches) > 1:
        raise ValueError(f"Multiple samples found for file name: {file_name}")

    row_index = int(matches[0])
    sample_embedding = embeddings[row_index : row_index + 1]
    scores = classifier.scores(sample_embedding)[0]
    predicted_label = int(np.argmax(scores))
    true_label = int(metadata.loc[row_index, "label"])

    return {
        "file_name": file_name,
        "true_label": true_label,
        "true_emotion": EMOTION_NAMES[true_label],
        "predicted_label": predicted_label,
        "predicted_emotion": EMOTION_NAMES[predicted_label],
        "scores": {
            emotion: float(scores[index])
            for index, emotion in enumerate(EMOTION_NAMES)
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one CREMA-D sample with a prototype clustering model."
    )
    parser.add_argument("--file-name", required=True, help="CREMA-D file name, e.g. 1001_DFA_ANG_XX.wav")
    parser.add_argument("--model-dir", required=True, help="Directory containing prototype centroids")
    parser.add_argument("--embedding-dir", required=True, help="Directory containing 128D normalized embeddings")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = score_sample_by_filename(
        file_name=args.file_name,
        model_dir=args.model_dir,
        embedding_dir=args.embedding_dir
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
