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