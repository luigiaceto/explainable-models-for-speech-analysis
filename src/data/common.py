from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def save_metadata(metadata: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    return output_path


def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required_columns = {"file_name", "emotion", "label"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata file {metadata_path} is missing columns: {sorted(missing_columns)}"
        )
    return metadata

@dataclass(frozen=True)
class FeaturePaths:
    feature_path: Path
    metadata_path: Path


def resolve_feature_paths(feature_dir: str | Path) -> FeaturePaths:
    feature_dir = Path(feature_dir)
    return FeaturePaths(
        feature_path=feature_dir / "features.npy",
        metadata_path=feature_dir / "metadata.csv"
    )


def load_features(
    feature_dir: str | Path,
    mmap_mode: str | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    paths = resolve_feature_paths(feature_dir)
    if not paths.feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {paths.feature_path}")
    if not paths.metadata_path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {paths.metadata_path}")

    features = np.load(paths.feature_path, mmap_mode=mmap_mode)
    metadata = load_metadata(paths.metadata_path)
    if len(features) != len(metadata):
        raise ValueError(
            f"Feature rows ({len(features)}) do not match metadata rows ({len(metadata)})"
        )
    return features, metadata


class AudioFeatureDataset(Dataset):
    """PyTorch dataset backed by precomputed pooled audio embeddings."""

    def __init__(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame,
        indices: Iterable[int] | None = None
    ) -> None:
        # load all features and labels at once
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(
            metadata["label"].to_numpy(dtype=np.int64),
            dtype=torch.long
        )
        self.metadata = metadata.reset_index(drop=True)
        self.indices = (
            np.arange(len(self.metadata), dtype=np.int64)
            if indices is None
            else np.asarray(list(indices), dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_index = int(self.indices[item])
        feature = self.features[row_index]
        label = self.labels[row_index]
        return feature, label
    
def make_feature_loader(
    features: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> DataLoader:
    """Create a DataLoader for one split of precomputed CREMA-D features."""
    indices = metadata.index[metadata["split"] == split_name].tolist()
    dataset = AudioFeatureDataset(
        features=features,
        metadata=metadata,
        indices=indices
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

