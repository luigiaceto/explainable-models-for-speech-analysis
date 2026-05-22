from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


EMOTION_CODE_TO_NAME = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

EMOTION_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

EMOTION_NAME_TO_LABEL = {
    name: index for index, name in enumerate(EMOTION_NAMES)
} # { "anger": 0, "disgust": 1, ... }

INTENSITY_CODE_TO_NAME = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified"
}

SENTENCE_CODE_TO_TEXT = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}


def parse_crema_d_filename(file_name: str) -> dict[str, object]:
    """Parse CREMA-D filenames such as ``1001_DFA_ANG_XX.wav``."""
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Expected CREMA-D filename with 4 parts, got: {file_name}")

    actor_id, sentence_code, emotion_code, intensity_code = parts
    if emotion_code not in EMOTION_CODE_TO_NAME:
        raise ValueError(f"Unknown emotion code '{emotion_code}' in: {file_name}")

    emotion = EMOTION_CODE_TO_NAME[emotion_code]
    return {
        "file_name": Path(file_name).name,
        "actor_id": actor_id,
        "sentence_code": sentence_code,
        "sentence": SENTENCE_CODE_TO_TEXT.get(sentence_code, "unknown"),
        "emotion_code": emotion_code,
        "emotion": emotion,
        "label": EMOTION_NAME_TO_LABEL[emotion],
        "intensity_code": intensity_code,
        "intensity": INTENSITY_CODE_TO_NAME.get(intensity_code, "unknown")
    }


def build_metadata_from_audio_dir(audio_dir: str | Path) -> pd.DataFrame:
    """Build a metadata table by scanning a CREMA-D ``AudioWAV`` directory."""
    audio_dir = Path(audio_dir)
    records = []
    for audio_path in sorted(audio_dir.glob("*.wav")):
        record = parse_crema_d_filename(audio_path.name)
        record["audio_path"] = str(audio_path)
        records.append(record)

    if not records:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    return pd.DataFrame(records).sort_values("file_name").reset_index(drop=True)


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


def emotion_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return sample counts and percentages for each emotion class."""
    counts = (
        metadata["emotion"]
        .value_counts()
        .reindex(EMOTION_NAMES, fill_value=0)
        .rename_axis("emotion")
        .reset_index(name="sample_count")
    )
    counts["percentage"] = counts["sample_count"] / len(metadata) * 100.0
    return counts


def print_dataset_statistics(metadata: pd.DataFrame) -> None:
    """Print compact CREMA-D statistics useful in notebooks and scripts."""
    print(f"Total samples: {len(metadata)}")
    if "actor_id" in metadata.columns:
        print(f"Actors: {metadata['actor_id'].nunique()}")
    if "sentence_code" in metadata.columns:
        print(f"Sentence prompts: {metadata['sentence_code'].nunique()}")
    print("\nSamples per emotion:")
    print(emotion_distribution(metadata).to_string(index=False))


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


class CremaDFeatureDataset(Dataset):
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


def make_crema_d_feature_loader(
    features: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> DataLoader:
    """Create a DataLoader for one split of precomputed CREMA-D features."""
    indices = metadata.index[metadata["split"] == split_name].tolist()
    dataset = CremaDFeatureDataset(
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
