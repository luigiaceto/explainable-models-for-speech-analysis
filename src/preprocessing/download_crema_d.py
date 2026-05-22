from __future__ import annotations
from pathlib import Path
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm.auto import tqdm
from src.data.crema_d import parse_crema_d_filename, save_metadata


# The manteiner of this HF dataset put all the crema-d dataset in the
# "train" split. We download the entire dataset. Then we will perform
# the embedding extraction using the audio encoder: the splits will be
# performed on the embedding dataset.
DEFAULT_DATASET_NAME = "cfahlgren1/crema-d"


def download_crema_d(
    output_dir: str | Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "train",
    sampling_rate: int = 16_000,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download the audio-only CREMA-D mirror from Hugging Face.

    The function writes WAV files to output_dir/AudioWAV and a normalized
    metadata table to output_dir/metadata.csv.
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "AudioWAV"
    metadata_path = output_dir / "metadata.csv"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # do not re-download the dataset if it has already been downloaded
    if metadata_path.exists() and not overwrite:
        metadata = pd.read_csv(metadata_path)
        expected_files = [audio_dir / file_name for file_name in metadata["file_name"]]
        if expected_files and all(path.exists() for path in expected_files):
            return metadata

    # load dataset from HuggingFace
    dataset = load_dataset(dataset_name, split=split)
    # get the audio and re-sample it
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    records = []
    for example in tqdm(dataset, desc="Writing CREMA-D WAV files"):
        file_name = example.get("source_file")
        if not file_name:
            audio_path = Path(example["audio"].get("path", ""))
            file_name = audio_path.name
        if not file_name:
            raise ValueError("Could not infer source filename from dataset example")

        # extract sample information from the audio name
        record = parse_crema_d_filename(file_name)
        target_path = audio_dir / record["file_name"]

        if overwrite or not target_path.exists():
            audio = example["audio"]
            sf.write(target_path, audio["array"], audio["sampling_rate"])

        record["audio_path"] = str(target_path)
        records.append(record)

    metadata = pd.DataFrame(records).sort_values("file_name").reset_index(drop=True)
    # saves metadata of the dataset as a CSV
    save_metadata(metadata, metadata_path)
    return metadata
