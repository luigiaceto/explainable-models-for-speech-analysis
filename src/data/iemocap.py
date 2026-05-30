from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


EMOTION_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
]

EMOTION_NAME_TO_LABEL = {
    name: index for index, name in enumerate(EMOTION_NAMES)
}

HF_EMOTION_ALIASES = {
    "ang": "angry",
    "angry": "angry",
    "hap": "happy",
    "happy": "happy",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sad",
}

IEMOCAP_FILENAME_PATTERN = re.compile(r"^(Ses\d{2})([FM])_")
IEMOCAP_SESSION_PATTERN = re.compile(r"^(?:Session|Ses)(\d+)$", re.IGNORECASE)


def normalize_emotion_name(emotion: str) -> str:
    """Normalize IEMOCAP emotion names to the project label vocabulary."""
    normalized = str(emotion).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = HF_EMOTION_ALIASES.get(normalized, normalized)
    if normalized not in EMOTION_NAME_TO_LABEL:
        raise ValueError(f"Unsupported IEMOCAP emotion label: {emotion!r}")
    return normalized


def normalize_session_id(session_name: str) -> str:
    """Normalize Hugging Face split names such as ``Session1`` to ``Ses01``."""
    match = IEMOCAP_SESSION_PATTERN.match(str(session_name))
    if match is None:
        return str(session_name)
    return f"Ses{int(match.group(1)):02d}"


def parse_iemocap_filename(file_name: str) -> dict[str, object]:
    """Parse IEMOCAP filenames such as ``Ses01F_impro01_F000.wav``."""
    file_name = Path(file_name).name
    match = IEMOCAP_FILENAME_PATTERN.match(Path(file_name).stem)
    if match is None:
        return {"file_name": file_name}

    session_id, speaker_gender_code = match.groups()
    speaker_id = f"{session_id}{speaker_gender_code}"
    return {
        "file_name": file_name,
        "session_id": session_id,
        "speaker_id": speaker_id,
        "speaker_gender_code": speaker_gender_code,
    }


def build_metadata_record(
    file_name: str,
    emotion: str,
    audio_path: str | Path,
    duration_seconds: float | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    normalized_emotion = normalize_emotion_name(emotion)
    record = parse_iemocap_filename(file_name)
    if session_id is not None:
        record["session_id"] = normalize_session_id(session_id)
    record.update(
        {
            "emotion": normalized_emotion,
            "label": EMOTION_NAME_TO_LABEL[normalized_emotion],
            "audio_path": str(audio_path),
        }
    )
    if duration_seconds is not None:
        record["duration_seconds"] = float(duration_seconds)
    columns = ["file_name", "session_id", "emotion", "label", "audio_path", "duration_seconds"]
    return {column: record[column] for column in columns if column in record}


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
    """Print compact IEMOCAP statistics useful in notebooks and scripts."""
    print(f"Total samples: {len(metadata)}")
    if "speaker_id" in metadata.columns:
        print(f"Speakers: {metadata['speaker_id'].nunique()}")
    if "session_id" in metadata.columns:
        print(f"Sessions: {metadata['session_id'].nunique()}")
    if "duration_seconds" in metadata.columns:
        total_duration_hours = metadata["duration_seconds"].sum() / 3600.0
        print(f"Audio duration: {total_duration_hours:.2f} hours")
        print(
            "Duration range: "
            f"{metadata['duration_seconds'].min():.2f}s - "
            f"{metadata['duration_seconds'].max():.2f}s"
        )
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
        metadata_path=feature_dir / "metadata.csv",
    )


def load_features(
    feature_dir: str | Path,
    mmap_mode: str | None = None,
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


class IemocapFeatureDataset(Dataset):
    """PyTorch dataset backed by precomputed pooled audio embeddings."""

    def __init__(
        self,
        features: np.ndarray,
        metadata: pd.DataFrame,
        indices: Iterable[int] | None = None,
    ) -> None:
        self.features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
        self.labels = torch.as_tensor(
            metadata["label"].to_numpy(dtype=np.int64),
            dtype=torch.long,
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


def make_iemocap_feature_loader(
    features: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for one split of precomputed IEMOCAP features."""
    indices = metadata.index[metadata["split"] == split_name].tolist()
    dataset = IemocapFeatureDataset(
        features=features,
        metadata=metadata,
        indices=indices,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
