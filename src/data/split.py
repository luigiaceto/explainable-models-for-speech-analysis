from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SAMPLE_STRATIFIED_SPLIT = "sample_stratified"
SPEAKER_INDEPENDENT_SPLIT = "speaker_independent"
SUPPORTED_SPLIT_STRATEGIES = {
    SAMPLE_STRATIFIED_SPLIT,
    SPEAKER_INDEPENDENT_SPLIT
}


def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total:.4f}")
    for split_name, split_size in {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }.items():
        if split_size <= 0.0:
            raise ValueError(f"{split_name} must be positive, got {split_size}")


def create_stratified_splits(
    metadata: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int
) -> pd.DataFrame:
    """Create stratified train/validation/test splits by emotion label."""
    metadata = metadata.reset_index(drop=True)
    _validate_split_sizes(train_size, val_size, test_size)

    indices = np.arange(len(metadata))
    labels = metadata["label"].to_numpy()

    # split from [dataset=train+val+test] to [train] and [val+test]
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=random_state,
        stratify=labels
    )
    # split from [val+test] to [val] and [test]
    relative_val_size = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_val_size,
        random_state=random_state,
        stratify=labels[temp_indices]
    )

    splits = metadata.copy()
    splits["split"] = ""
    splits.loc[train_indices, "split"] = "train"
    splits.loc[val_indices, "split"] = "val"
    splits.loc[test_indices, "split"] = "test"
    splits["split_strategy"] = SAMPLE_STRATIFIED_SPLIT
    return splits


def create_speaker_independent_splits(
    metadata: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
    speaker_column: str = "actor_id"
) -> pd.DataFrame:
    """Create train/validation/test splits with disjoint speakers."""
    metadata = metadata.reset_index(drop=True)
    _validate_split_sizes(train_size, val_size, test_size)
    if speaker_column not in metadata.columns:
        raise ValueError(
            f"Cannot create speaker-independent split: missing '{speaker_column}' column"
        )

    speakers = metadata[speaker_column].drop_duplicates().to_numpy()

    train_speakers, temp_speakers = train_test_split(
        speakers,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )
    relative_val_size = val_size / (val_size + test_size)
    val_speakers, test_speakers = train_test_split(
        temp_speakers,
        train_size=relative_val_size,
        random_state=random_state,
        shuffle=True
    )

    splits = metadata.copy()
    splits["split"] = ""
    splits.loc[splits[speaker_column].isin(train_speakers), "split"] = "train"
    splits.loc[splits[speaker_column].isin(val_speakers), "split"] = "val"
    splits.loc[splits[speaker_column].isin(test_speakers), "split"] = "test"
    splits["split_strategy"] = SPEAKER_INDEPENDENT_SPLIT
    _validate_speaker_independent_split(splits, speaker_column)
    return splits


def _validate_speaker_independent_split(
    metadata: pd.DataFrame,
    speaker_column: str
) -> None:
    missing_split_count = int((metadata["split"] == "").sum())
    if missing_split_count:
        raise ValueError(f"{missing_split_count} samples were not assigned to any split")

    speakers_by_split = {
        split_name: set(metadata.loc[metadata["split"] == split_name, speaker_column])
        for split_name in ("train", "val", "test")
    }
    for first_split, second_split in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = speakers_by_split[first_split].intersection(speakers_by_split[second_split])
        if overlap:
            raise ValueError(
                f"Speaker leakage between {first_split} and {second_split}: "
                f"{sorted(overlap)}"
            )


def create_splits(
    metadata: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
    split_strategy: str,
    speaker_column: str = "actor_id"
) -> pd.DataFrame:
    """Create train/validation/test splits using the selected strategy."""
    if split_strategy == SAMPLE_STRATIFIED_SPLIT:
        return create_stratified_splits(
            metadata=metadata,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state
        )
    if split_strategy == SPEAKER_INDEPENDENT_SPLIT:
        return create_speaker_independent_splits(
            metadata=metadata,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            speaker_column=speaker_column
        )
    raise ValueError(
        f"Unsupported split strategy '{split_strategy}'. "
        f"Choose one of: {sorted(SUPPORTED_SPLIT_STRATEGIES)}"
    )
