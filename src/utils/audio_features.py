from __future__ import annotations


def pooled_feature_dim(encoder_embedding_dim: int, pooling: str) -> int:
    normalized_pooling = pooling.lower()
    if normalized_pooling == "mean_std":
        return encoder_embedding_dim * 2
    if normalized_pooling == "mean":
        return encoder_embedding_dim
    raise ValueError(f"Unsupported pooling: {pooling}")
