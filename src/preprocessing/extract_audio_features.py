from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from src.data.iemocap import load_metadata, resolve_feature_paths
from src.utils.audio_features import pooled_feature_dim
from src.utils.naming import model_name_to_slug
from src.utils.utils import device_or_default


DEFAULT_MODEL_NAME = "microsoft/wavlm-base-plus"
DEFAULT_POOLING = "mean_std"


def _load_waveform(
    audio_path: Path,
    target_sampling_rate: int,
    resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] | None = None
) -> np.ndarray:
    waveform, sampling_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sampling_rate != target_sampling_rate:
        resampler_key = (sampling_rate, target_sampling_rate)
        if resamplers is None:
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        else:
            if resampler_key not in resamplers:
                resamplers[resampler_key] = torchaudio.transforms.Resample(
                    sampling_rate,
                    target_sampling_rate
                )
            resampler = resamplers[resampler_key]
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()


class _AudioWaveformDataset(Dataset):
    """Dataset that loads and normalizes raw waveforms for feature extraction."""

    def __init__(self, audio_paths: list[str], sampling_rate: int) -> None:
        self.audio_paths = audio_paths
        self.sampling_rate = sampling_rate
        self.resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, item: int) -> np.ndarray:
        return _load_waveform(
            Path(self.audio_paths[item]),
            self.sampling_rate,
            self.resamplers
        )


def _collate_waveforms(waveforms: list[np.ndarray]) -> list[np.ndarray]:
    """Keep variable-length waveforms as a list for Hugging Face padding."""
    return waveforms


def _feature_attention_mask(
    model: torch.nn.Module,
    attention_mask: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    if hasattr(model, "_get_feature_vector_attention_mask"):
        return model._get_feature_vector_attention_mask(sequence_length, attention_mask)

    input_lengths = attention_mask.sum(dim=1)
    if hasattr(model, "_get_feat_extract_output_lengths"):
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
    else:
        # estimate
        scale = sequence_length / attention_mask.shape[1]
        output_lengths = torch.ceil(input_lengths.float() * scale).long()

    positions = torch.arange(sequence_length, device=attention_mask.device)
    return positions.unsqueeze(0) < output_lengths.unsqueeze(1)


def _masked_mean_std_pooling(
    hidden_states: torch.Tensor,
    feature_mask: torch.Tensor
) -> torch.Tensor:
    mask = feature_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    mean = (hidden_states * mask).sum(dim=1) / denominator
    variance = ((hidden_states - mean.unsqueeze(1)).pow(2) * mask).sum(dim=1) / denominator
    std = torch.sqrt(variance.clamp(min=1e-8))
    return torch.cat([mean, std], dim=-1)


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    feature_mask: torch.Tensor,
    pooling: str
) -> torch.Tensor:
    normalized_pooling = pooling.lower()
    if normalized_pooling == "mean_std":
        return _masked_mean_std_pooling(hidden_states, feature_mask)
    raise ValueError(f"Unsupported pooling: {pooling}")


def _metadata_with_audio_paths(metadata: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    metadata = metadata.copy()
    if "audio_path" not in metadata.columns:
        metadata["audio_path"] = metadata["file_name"].map(lambda name: str(audio_dir / name))
    metadata["audio_path"] = metadata["audio_path"].map(
        lambda path: str(Path(path)) if Path(path).is_absolute() else str(audio_dir / Path(path).name)
    )
    missing = [path for path in metadata["audio_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing audio files, first missing path: {missing[0]}")
    return metadata


def _audio_duration_seconds(audio_path: str | Path) -> float:
    info = torchaudio.info(str(audio_path))
    if info.sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for audio file: {audio_path}")
    return float(info.num_frames / info.sample_rate)


def _metadata_filtered_by_duration(
    metadata: pd.DataFrame,
    min_duration_seconds: float | None,
    max_duration_seconds: float | None
) -> tuple[pd.DataFrame, int, int]:
    if min_duration_seconds is not None and min_duration_seconds < 0.0:
        raise ValueError(
            f"min_duration_seconds must be non-negative, got {min_duration_seconds}"
        )
    if max_duration_seconds is not None and max_duration_seconds <= 0.0:
        raise ValueError(
            f"max_duration_seconds must be positive, got {max_duration_seconds}"
        )
    if (
        min_duration_seconds is not None
        and max_duration_seconds is not None
        and max_duration_seconds <= min_duration_seconds
    ):
        raise ValueError(
            "max_duration_seconds must be greater than min_duration_seconds, "
            f"got {max_duration_seconds} <= {min_duration_seconds}"
        )

    metadata = metadata.copy()
    if "duration_seconds" not in metadata.columns:
        metadata["duration_seconds"] = metadata["audio_path"].map(_audio_duration_seconds)

    keep_mask = pd.Series(True, index=metadata.index)
    if min_duration_seconds is not None:
        keep_mask &= metadata["duration_seconds"] > min_duration_seconds
    if max_duration_seconds is not None:
        keep_mask &= metadata["duration_seconds"] <= max_duration_seconds

    kept_metadata = metadata[keep_mask].copy()
    removed_short_count = int(
        (metadata["duration_seconds"] <= min_duration_seconds).sum()
        if min_duration_seconds is not None
        else 0
    )
    removed_long_count = int(
        (metadata["duration_seconds"] > max_duration_seconds).sum()
        if max_duration_seconds is not None
        else 0
    )
    if kept_metadata.empty:
        raise ValueError(
            "No audio samples remain after duration filtering "
            f"(min_duration_seconds={min_duration_seconds}, "
            f"max_duration_seconds={max_duration_seconds})"
        )
    return kept_metadata.reset_index(drop=True), removed_short_count, removed_long_count


def _metadata_sorted_by_duration(metadata: pd.DataFrame) -> pd.DataFrame:
    if "duration_seconds" not in metadata.columns:
        raise ValueError("Cannot sort by duration: missing 'duration_seconds' column")
    return (
        metadata
        .sort_values(["duration_seconds", "file_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def _validate_existing_duration_filter(
    config_path: Path,
    min_duration_seconds: float | None,
    max_duration_seconds: float | None
) -> None:
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    existing_min_duration = config.get("min_duration_seconds")
    existing_max_duration = config.get("max_duration_seconds")
    if (
        existing_min_duration != min_duration_seconds
        or existing_max_duration != max_duration_seconds
    ):
        raise ValueError(
            "Existing features were extracted with different duration filters: "
            f"min={existing_min_duration}, max={existing_max_duration}. "
            "Use overwrite=True or a different output_dir to regenerate them."
        )


def _validate_expected_encoder_embedding_dim(
    model_config: Any,
    expected_encoder_embedding_dim: int | None
) -> None:
    if expected_encoder_embedding_dim is None:
        return

    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        return

    if int(hidden_size) != expected_encoder_embedding_dim:
        raise ValueError(
            f"Expected encoder embedding dim {expected_encoder_embedding_dim}, "
            f"but model hidden size is {hidden_size}"
        )


def extract_audio_features(
    metadata_csv: str | Path,
    audio_dir: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    expected_encoder_embedding_dim: int | None = None,
    pooling: str = DEFAULT_POOLING,
    batch_size: int = 8,
    sampling_rate: int = 16_000,
    device: str | None = None,
    overwrite: bool = False,
    num_workers: int = 0,
    min_duration_seconds: float | None = 2.0,
    max_duration_seconds: float | None = 15.0
) -> dict[str, Path]:
    """Extract pooled embeddings from a frozen audio encoder."""
    metadata_csv = Path(metadata_csv)
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_feature_paths(output_dir)
    config_path = output_dir / "feature_config.json"
    expected_pooled_feature_dim = (
        None
        if expected_encoder_embedding_dim is None
        else pooled_feature_dim(expected_encoder_embedding_dim, pooling)
    )

    if paths.feature_path.exists() and paths.metadata_path.exists() and not overwrite:
        _validate_existing_duration_filter(
            config_path,
            min_duration_seconds,
            max_duration_seconds
        )
        if expected_pooled_feature_dim is not None:
            existing_features = np.load(paths.feature_path, mmap_mode="r")
            if existing_features.shape[1] != expected_pooled_feature_dim:
                raise ValueError(
                    f"Expected pooled feature dim {expected_pooled_feature_dim}, "
                    f"but existing features have dim {existing_features.shape[1]}: "
                    f"{paths.feature_path}"
                )
        return {
            "features": paths.feature_path,
            "metadata": paths.metadata_path,
            "config": config_path,
        }

    source_metadata = _metadata_with_audio_paths(load_metadata(metadata_csv), audio_dir)
    metadata, removed_short_audio_count, removed_long_audio_count = _metadata_filtered_by_duration(
        source_metadata,
        min_duration_seconds,
        max_duration_seconds
    )
    metadata = _metadata_sorted_by_duration(metadata)
    compute_device = device_or_default(device)

    # Audio preprocessor for the encoder: normalizes raw waveforms and pads batches.
    # It prepares the audio to be fed to the encoder model.
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # audio encoder which produces embeddings from the input audio
    model = AutoModel.from_pretrained(model_name).to(compute_device)
    _validate_expected_encoder_embedding_dim(model.config, expected_encoder_embedding_dim)

    model_hidden_size = getattr(model.config, "hidden_size", None)
    encoder_embedding_dim = (
        int(model_hidden_size)
        if model_hidden_size is not None
        else expected_encoder_embedding_dim
    )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    audio_dataset = _AudioWaveformDataset(
        audio_paths=metadata["audio_path"].tolist(),
        sampling_rate=sampling_rate
    )
    audio_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_waveforms
    )

    pooled_batches = []
    model_slug = model_name_to_slug(model_name)
    for waveforms in tqdm(audio_loader, desc=f"Extracting {model_slug} features"):
        # The attention mask marks real audio samples versus padding.
        inputs = feature_extractor(
            waveforms,
            sampling_rate=sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        inputs = {key: value.to(compute_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            # the attention_mask is referred to the original waveform audio input, so its length
            # is equal to the number of audio samples post-padding. But the feature extractor
            # produces a hidden-state sequence much shorter than the number of audio samples.
            # We have to "translate" the mask of the audio to a mask of the hidden states.
            # We need this translated mask otherwise during the pooling process we would include
            # also hidden states obtained from padded audio section which is useless -> we want to
            # pool only valid audio frames
            if "attention_mask" in inputs:
                feature_mask = _feature_attention_mask(
                    model=model,
                    attention_mask=inputs["attention_mask"],
                    sequence_length=hidden_states.shape[1]
                )
            else:
                feature_mask = torch.ones(
                    hidden_states.shape[:2],
                    dtype=torch.bool,
                    device=hidden_states.device
                )
            pooled = _pool_hidden_states(hidden_states, feature_mask, pooling)
        pooled_batches.append(pooled.cpu().numpy().astype(np.float32))

    features = np.concatenate(pooled_batches, axis=0)
    if expected_pooled_feature_dim is not None and features.shape[1] != expected_pooled_feature_dim:
        raise ValueError(
            f"Expected pooled feature dim {expected_pooled_feature_dim}, got {features.shape[1]}"
        )

    np.save(paths.feature_path, features)
    metadata.to_csv(paths.metadata_path, index=False)

    config = {
        "model_name": model_name,
        "model_slug": model_slug,
        "pooling": pooling,
        "sampling_rate": sampling_rate,
        "encoder_embedding_dim": encoder_embedding_dim,
        "feature_dim": int(features.shape[1]),
        "feature_shape": list(features.shape),
        "num_workers": num_workers,
        "min_duration_seconds": min_duration_seconds,
        "max_duration_seconds": max_duration_seconds,
        "source_num_samples": int(len(source_metadata)),
        "filtered_num_samples": int(len(metadata)),
        "removed_short_audio_count": int(removed_short_audio_count),
        "removed_long_audio_count": int(removed_long_audio_count),
        "feature_extraction_order": "duration_seconds_ascending",
        "source_metadata": str(metadata_csv),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "features": paths.feature_path,
        "metadata": paths.metadata_path,
        "config": config_path,
    }
