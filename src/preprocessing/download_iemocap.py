from __future__ import annotations
from numbers import Integral
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import Audio, ClassLabel, load_dataset
from tqdm.auto import tqdm
from src.data.iemocap import (
    build_metadata_record,
    save_metadata,
)


DEFAULT_DATASET_NAME = "tarasabkar/IEMOCAP_Speech"


def _emotion_to_name(emotion: object, class_label: ClassLabel | None) -> str:
    if class_label is not None and isinstance(emotion, Integral):
        return str(class_label.int2str(emotion))
    return str(emotion)


def _all_dataset_items(dataset_name: str, split: str | None):
    dataset = load_dataset(dataset_name)
    if split is not None:
        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not found in {dataset_name}. "
                f"Available splits are: {list(dataset)}"
            )
        return [(split, dataset[split])]
    return list(dataset.items())


def download_iemocap(
    output_dir: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str | None = None,
    sampling_rate: int = 16_000,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download the 4-class preprocessed IEMOCAP speech mirror from Hugging Face.

    The function writes every WAV file to output_dir/audio and a normalized
    metadata table to output_dir/metadata.csv.
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

    records = []
    for split_name, split_dataset in _all_dataset_items(dataset_name, split):
        split_dataset = split_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
        emotion_feature = split_dataset.features.get("emotion")
        class_label = emotion_feature if isinstance(emotion_feature, ClassLabel) else None

        for index, example in enumerate(
            tqdm(split_dataset, desc=f"Writing IEMOCAP WAV files ({split_name})")
        ):
            audio = example["audio"]
            file_name = f"{split_name}_{index:05d}.wav"
            target_path = audio_dir / file_name
            if overwrite or not target_path.exists():
                sf.write(target_path, audio["array"], audio["sampling_rate"])

            duration_seconds = float(len(audio["array"]) / audio["sampling_rate"])
            emotion = _emotion_to_name(example["emotion"], class_label)
            record = build_metadata_record(
                file_name=file_name,
                emotion=emotion,
                audio_path=target_path,
                duration_seconds=duration_seconds,
                session_id=split_name,
            )
            records.append(record)

    metadata = pd.DataFrame(records).sort_values("file_name").reset_index(drop=True)
    save_metadata(metadata, metadata_path)
    return metadata
