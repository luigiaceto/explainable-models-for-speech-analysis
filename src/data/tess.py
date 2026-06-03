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
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad",
]

EMOTION_NAME_TO_LABEL = {
    name: index for index, name in enumerate(EMOTION_NAMES)
}

HF_EMOTION_ALIASES = {
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "pleasant_surprise",
    "pleasant surprise": "pleasant_surprise",
    "pleasant-surprise": "pleasant_surprise",
    "ps": "pleasant_surprise",
    "sad": "sad",
}

TESS_FILENAME_PATTERN = re.compile(
    r"^(?P<speaker_id>[OY]AF)_(?P<word>.+)_(?P<emotion>angry|disgust|fear|happy|neutral|ps|sad)$",
    re.IGNORECASE,
)
TESS_SPEAKER_GROUPS = {
    "OAF": "older_adult_female",
    "YAF": "younger_adult_female",
}


def normalize_emotion_name(emotion: str) -> str:
    """Normalize TESS emotion names to the project label vocabulary."""
    normalized = str(emotion).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = HF_EMOTION_ALIASES.get(normalized, normalized)
    if normalized not in EMOTION_NAME_TO_LABEL:
        raise ValueError(f"Unsupported TESS emotion label: {emotion!r}")
    return normalized


def parse_tess_filename(file_name: str) -> dict[str, object]:
    """Parse TESS filenames such as ``OAF_back_angry.wav``."""
    file_name = Path(file_name).name
    match = TESS_FILENAME_PATTERN.match(Path(file_name).stem)
    if match is None:
        return {"file_name": file_name}

    speaker_id = match.group("speaker_id").upper()
    word = match.group("word").lower()
    file_emotion = normalize_emotion_name(match.group("emotion"))
    return {
        "file_name": file_name,
        "speaker_id": speaker_id,
        "speaker_group": TESS_SPEAKER_GROUPS.get(speaker_id, speaker_id.lower()),
        "speaker_gender": "female",
        "word": word,
        "file_emotion": file_emotion,
    }


def build_metadata_record(
    file_name: str,
    emotion: str,
    audio_path: str | Path,
    duration_seconds: float | None = None,
    gender: str | None = None,
    transcription: str | None = None,
) -> dict[str, object]:
    normalized_emotion = normalize_emotion_name(emotion)
    record = parse_tess_filename(file_name)
    if "file_emotion" in record and record["file_emotion"] != normalized_emotion:
        raise ValueError(
            f"File name emotion {record['file_emotion']!r} does not match "
            f"metadata emotion {normalized_emotion!r} for {file_name}"
        )
    if gender is not None:
        record["speaker_gender"] = str(gender).strip().lower()
    if transcription is not None:
        record["transcription"] = str(transcription)
    record.update(
        {
            "emotion": normalized_emotion,
            "label": EMOTION_NAME_TO_LABEL[normalized_emotion],
            "audio_path": str(audio_path),
        }
    )
    if duration_seconds is not None:
        record["duration_seconds"] = float(duration_seconds)
    columns = [
        "file_name",
        "speaker_id",
        "speaker_group",
        "speaker_gender",
        "word",
        "transcription",
        "emotion",
        "label",
        "audio_path",
        "duration_seconds",
    ]
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
    """Print compact TESS statistics useful in notebooks and scripts."""
    print(f"Total samples: {len(metadata)}")
    if "speaker_id" in metadata.columns:
        print(f"Speakers: {metadata['speaker_id'].nunique()}")
    if "word" in metadata.columns:
        print(f"Words: {metadata['word'].nunique()}")
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


class TessFeatureDataset(Dataset):
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


def make_tess_feature_loader(
    features: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for one split of precomputed TESS features."""
    indices = metadata.index[metadata["split"] == split_name].tolist()
    dataset = TessFeatureDataset(
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
