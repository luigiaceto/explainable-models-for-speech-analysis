from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from src.data.crema_d import load_features, resolve_feature_paths
from src.evaluation.evaluate_blackbox import load_blackbox_model
from src.models.prototype_clustering import l2_normalize_rows


def _load_aligned_split_metadata(
    feature_metadata: pd.DataFrame,
    splits_csv: str | Path
) -> pd.DataFrame:
    split_metadata = pd.read_csv(splits_csv)
    if len(split_metadata) != len(feature_metadata):
        raise ValueError("Split metadata and feature metadata have different lengths")
    if feature_metadata["file_name"].tolist() != split_metadata["file_name"].tolist():
        raise ValueError("Split metadata and feature metadata are not aligned")
    return split_metadata


def extract_blackbox_penultimate_embeddings(
    feature_dir: str | Path,
    checkpoint_path: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path,
    batch_size: int = 256,
    device: str | None = None,
    overwrite: bool = False
) -> dict[str, Path]:
    """Extract L2-normalized 128D black-box penultimate embeddings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_feature_paths(output_dir)
    config_path = output_dir / "embedding_config.json"

    if paths.feature_path.exists() and paths.metadata_path.exists() and not overwrite:
        return {
            "features": paths.feature_path,
            "metadata": paths.metadata_path,
            "config": config_path,
        }

    features, feature_metadata = load_features(feature_dir, mmap_mode="r")
    split_metadata = _load_aligned_split_metadata(feature_metadata, splits_csv)
    model, checkpoint, compute_device = load_blackbox_model(checkpoint_path, device)
    penultimate_network = model.network[:-1]

    embedding_batches = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(features), batch_size),
            desc="Extracting black-box penultimate embeddings"
        ):
            batch = torch.as_tensor(
                np.asarray(features[start : start + batch_size], dtype=np.float32),
                dtype=torch.float32,
                device=compute_device
            )
            embeddings = penultimate_network(batch).cpu().numpy().astype(np.float32)
            embedding_batches.append(embeddings)

    embeddings = np.concatenate(embedding_batches, axis=0)
    normalized_embeddings = l2_normalize_rows(embeddings)

    np.save(paths.feature_path, normalized_embeddings.astype(np.float32))
    split_metadata.to_csv(paths.metadata_path, index=False)

    blackbox_config: dict[str, Any] = checkpoint["config"]
    config = {
        "source_feature_dir": str(feature_dir),
        "source_checkpoint": str(checkpoint_path),
        "source_splits": str(splits_csv),
        "embedding_dim": int(normalized_embeddings.shape[1]),
        "embedding_shape": list(normalized_embeddings.shape),
        "normalization": "l2",
        "source_model": blackbox_config.get("feature_extractor_name"),
        "source_pooling": blackbox_config.get("pooling"),
        "source_hidden_dims": blackbox_config.get("hidden_dims"),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "features": paths.feature_path,
        "metadata": paths.metadata_path,
        "config": config_path,
    }
