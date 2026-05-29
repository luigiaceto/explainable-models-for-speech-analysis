from __future__ import annotations
import subprocess
import urllib.request
import tarfile
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from src.data.meld import EMOTION_NAME_TO_LABEL
from src.data.common import save_metadata 
from src.utils.utils import download_with_progress, extract_tar_gz


# Official MELD data URLs
RAW_DATA_URL = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
CSV_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"
}

def setup_raw_data(raw_data_dir: Path) -> None:
    """Downloads the MELD CSVs and MP4 archives, and extracts them."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download CSVs
    for split_name, url in CSV_URLS.items():
        csv_path = raw_data_dir / f"{split_name}_sent_emo.csv"
        download_with_progress(url, csv_path, f"Downloading {split_name} CSV")

    # 2. Download Raw Master Tar (Videos)
    print("\nConnecting to Hugging Face to download MELD.Raw.tar.gz (11.8 GB)...")
    master_tar_path = hf_hub_download(
        repo_id="declare-lab/MELD",
        repo_type="dataset",
        filename="MELD.Raw.tar.gz",
        local_dir=raw_data_dir  # Tell it to save directly into your raw data folder
    )
    
    master_tar_path = Path(master_tar_path)

    # 3. Extract Master Tar
    # The master tar usually unpacks directly into the directory or creates a subfolder.
    extract_tar_gz(master_tar_path, raw_data_dir)

    # 4. Extract nested sub-archives (train.tar.gz, dev.tar.gz, test.tar.gz)
    for split in ["train", "dev", "test"]:
        sub_tar = raw_data_dir / f"{split}.tar.gz"
        if sub_tar.exists():
            extract_tar_gz(sub_tar, raw_data_dir)
            # Optional: delete the sub_tar to save disk space in Colab
            sub_tar.unlink()


def extract_audio_from_video(video_path: Path, audio_path: Path, sampling_rate: int) -> bool:
    """Extracts mono audio from an mp4 file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sampling_rate),
        "-ac", "1",
        str(audio_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def download_and_process_meld(
    output_dir: str | Path,
    sampling_rate: int = 16_000,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Downloads raw MELD data, extracts audio, and builds the metadata.csv.
    """
    output_dir = Path(output_dir)
    raw_data_dir = output_dir / "raw_downloads"
    audio_dir = output_dir / "AudioWAV"
    metadata_path = output_dir / "metadata.csv"

    # Step 1: Download and unpack everything
    setup_raw_data(raw_data_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)

    # MELD split mappings: (CSV name, Extracted Folder Name, Pipeline Split Name)
    splits_to_process = [
        ("train_sent_emo.csv", "train", "train"),
        ("dev_sent_emo.csv", "dev", "val"),
        ("test_sent_emo.csv", "test", "test"),
    ]

    records = []
    
    # Step 2: Convert MP4s to WAVs and build metadata
    for csv_name, video_folder, split_name in splits_to_process:
        csv_path = raw_data_dir / csv_name
        video_dir = raw_data_dir / video_folder
        
        if not csv_path.exists() or not video_dir.exists():
            print(f"Warning: Missing data for {split_name} split. Skipping.")
            continue
            
        df = pd.read_csv(csv_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing MELD {split_name}"):
            file_stem = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            video_path = video_dir / f"{file_stem}.mp4"
            audio_path = audio_dir / f"{file_stem}.wav"
            
            emotion = str(row["Emotion"]).lower()
            if emotion not in EMOTION_NAME_TO_LABEL:
                continue 
                
            if overwrite or not audio_path.exists():
                if not video_path.exists():
                    continue 
                success = extract_audio_from_video(video_path, audio_path, sampling_rate)
                if not success:
                    continue

            records.append({
                "file_name": audio_path.name,
                "audio_path": str(audio_path),
                "split": split_name,
                "emotion": emotion,
                "label": EMOTION_NAME_TO_LABEL[emotion],
                "sentiment": str(row["Sentiment"]).lower(),
                "Speaker": row["Speaker"],
                "Dialogue_ID": row["Dialogue_ID"],
                "Utterance_ID": row["Utterance_ID"],
                "text": row["Utterance"]
            })

    if not records:
        raise RuntimeError("No MELD records were processed.")

    metadata = pd.DataFrame(records).sort_values("file_name").reset_index(drop=True)
    save_metadata(metadata, metadata_path)
    
    print(f"Successfully processed {len(metadata)} MELD files.")
    return metadata    