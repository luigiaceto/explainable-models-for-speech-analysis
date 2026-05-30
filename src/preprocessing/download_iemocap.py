from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm.auto import tqdm
from src.data.iemocap import (
    EMOTION_SCORE_COLUMNS,
    build_metadata_record,
    save_metadata,
)


DEFAULT_DATASET_NAME = "AbstractTTS/IEMOCAP"


def _safe_extra_fields(example: dict[str, object]) -> dict[str, object]:
    extra_fields: dict[str, object] = {}
    for column in (
        "gender",
        "transcription",
        "major_emotion",
        "EmoAct",
        "EmoVal",
        "EmoDom",
        "speaking_rate",
        "pitch_mean",
        "pitch_std",
        "rms",
        "relative_db",
    ):
        if column in example and example[column] is not None:
            extra_fields[column] = example[column]

    for column in EMOTION_SCORE_COLUMNS:
        if column in example and example[column] is not None:
            extra_fields[f"{column}_score"] = example[column]
    return extra_fields


def download_iemocap(
    output_dir: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "train",
    sampling_rate: int = 16_000,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download the audio-only IEMOCAP mirror from Hugging Face.

    The function writes every WAV file to output_dir/audio and a normalized
    metadata table to output_dir/metadata.csv. Short audio filtering is applied
    later during frozen audio encoder feature extraction.
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    metadata_path = output_dir / "metadata.csv"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and not overwrite:
        metadata = pd.read_csv(metadata_path)
        expected_files = [audio_dir / file_name for file_name in metadata["file_name"]]
        if expected_files and all(path.exists() for path in expected_files):
            return metadata

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    records = []
    for example in tqdm(dataset, desc="Writing IEMOCAP WAV files"):
        file_name = example.get("file")
        if not file_name:
            audio_path = Path(example["audio"].get("path", ""))
            file_name = audio_path.name
        if not file_name:
            raise ValueError("Could not infer source filename from dataset example")

        emotion = example.get("major_emotion")
        if emotion is None:
            raise ValueError(f"Missing major_emotion for sample {file_name}")

        audio = example["audio"]
        target_path = audio_dir / Path(str(file_name)).name
        if overwrite or not target_path.exists():
            sf.write(target_path, audio["array"], audio["sampling_rate"])

        duration_seconds = float(len(audio["array"]) / audio["sampling_rate"])
        record = build_metadata_record(
            file_name=str(file_name),
            emotion=str(emotion),
            audio_path=target_path,
            duration_seconds=duration_seconds,
            extra_fields=_safe_extra_fields(example),
        )
        records.append(record)

    metadata = pd.DataFrame(records).sort_values("file_name").reset_index(drop=True)
    save_metadata(metadata, metadata_path)
    return metadata
