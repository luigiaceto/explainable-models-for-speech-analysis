from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from src.data.crema_d import load_metadata, resolve_feature_paths
from src.utils.utils import device_or_default


DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"


def _load_waveform(audio_path: Path, target_sampling_rate: int) -> np.ndarray:
    waveform, sampling_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()


def _feature_attention_mask(
    model: Wav2Vec2Model,
    attention_mask: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    if hasattr(model, "_get_feature_vector_attention_mask"):
        return model._get_feature_vector_attention_mask(sequence_length, attention_mask)

    input_lengths = attention_mask.sum(dim=1)
    if hasattr(model, "_get_feat_extract_output_lengths"):
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
    else:
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


def extract_wav2vec_features(
    metadata_csv: str | Path,
    audio_dir: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
    sampling_rate: int = 16_000,
    device: str | None = None,
    overwrite: bool = False
) -> dict[str, Path]:
    """Extract mean+std pooled wav2vec2 embeddings for every CREMA-D clip."""
    metadata_csv = Path(metadata_csv)
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_feature_paths(output_dir)
    config_path = output_dir / "feature_config.json"

    # prevent repeating extraction if it has already been made
    if paths.feature_path.exists() and paths.metadata_path.exists() and not overwrite:
        return {
            "features": paths.feature_path,
            "metadata": paths.metadata_path,
            "config": config_path,
        }

    metadata = _metadata_with_audio_paths(load_metadata(metadata_csv), audio_dir)
    compute_device = device_or_default(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(compute_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    pooled_batches = []
    for start in tqdm(range(0, len(metadata), batch_size), desc="Extracting wav2vec2 features"):
        batch = metadata.iloc[start : start + batch_size]
        waveforms = [
            _load_waveform(Path(audio_path), sampling_rate)
            for audio_path in batch["audio_path"].tolist()
        ]
        # attention mask is used with the audios since padding is applied in order to have valid
        # torch batches. Attention mask indicates which audio parts are real and which
        # are padding
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
            # produces a sequence of hiddens states way smaller than the number of audio samples.
            # We have to "translate" the mask of the audio to a mask of the hidden states.
            # We need this translated mask otherwise during the pooling process we would include
            # also hidden states obtained from padded audio section which is useless -> we want to
            # pool only valid audio frames
            feature_mask = _feature_attention_mask(
                model=model,
                attention_mask=inputs["attention_mask"],
                sequence_length=hidden_states.shape[1]
            )
            pooled = _masked_mean_std_pooling(hidden_states, feature_mask)
        pooled_batches.append(pooled.cpu().numpy().astype(np.float32))

    features = np.concatenate(pooled_batches, axis=0)
    np.save(paths.feature_path, features)
    metadata.to_csv(paths.metadata_path, index=False)

    config = {
        "model_name": model_name,
        "pooling": "masked_mean_std",
        "sampling_rate": sampling_rate,
        "feature_shape": list(features.shape),
        "source_metadata": str(metadata_csv),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "features": paths.feature_path,
        "metadata": paths.metadata_path,
        "config": config_path,
    }
