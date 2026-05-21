from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.data.crema_d import EMOTION_NAMES, load_features
from src.evaluation.evaluate_blackbox import load_blackbox_model


def _pca_projection(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    random_state: int
) -> tuple[pd.DataFrame, PCA]:
    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=2, random_state=random_state)
    coordinates = pca.fit_transform(scaled_embeddings)

    projection = metadata[["file_name", "emotion", "label", "split"]].copy()
    projection["pca_1"] = coordinates[:, 0]
    projection["pca_2"] = coordinates[:, 1]
    return projection, pca


def _blackbox_penultimate_embeddings(
    embeddings: np.ndarray,
    checkpoint_path: str | Path,
    batch_size: int,
    device: str | None
) -> np.ndarray:
    model, _, compute_device = load_blackbox_model(checkpoint_path, device)
    penultimate_network = model.network[:-1]
    penultimate_batches = []

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.as_tensor(
                embeddings[start : start + batch_size],
                dtype=torch.float32,
                device=compute_device
            )
            penultimate_batches.append(penultimate_network(batch).cpu().numpy())

    return np.concatenate(penultimate_batches, axis=0)


def plot_blackbox_embedding_pca(
    feature_dir: str | Path,
    checkpoint_path: str | Path,
    splits_csv: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    sample_limit: int | None = None,
    random_state: int = 42,
    batch_size: int = 256,
    device: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (15, 6),
    dpi: int = 180
) -> dict[str, Any]:
    """Plot two-dimensional PCA projections of black-box embedding spaces.

    The first panel shows the frozen wav2vec2 mean+std pooled features. The
    second panel shows the trained black-box representation before the final
    classification layer.
    """
    features, feature_metadata = load_features(feature_dir)
    split_metadata = pd.read_csv(splits_csv)

    if feature_metadata["file_name"].tolist() != split_metadata["file_name"].tolist():
        raise ValueError("Feature metadata and split metadata are not aligned.")

    if split == "all":
        plot_indices = np.arange(len(split_metadata))
    else:
        plot_indices = split_metadata.index[split_metadata["split"] == split].to_numpy()

    if len(plot_indices) == 0:
        raise ValueError(f"No samples found for split '{split}'.")

    if sample_limit is not None and len(plot_indices) > sample_limit:
        rng = np.random.default_rng(random_state)
        plot_indices = np.sort(rng.choice(plot_indices, size=sample_limit, replace=False))

    plot_metadata = split_metadata.iloc[plot_indices].reset_index(drop=True)
    wav2vec_embeddings = np.asarray(features[plot_indices], dtype=np.float32)
    mlp_embeddings = _blackbox_penultimate_embeddings(
        embeddings=wav2vec_embeddings,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device
    )

    wav2vec_projection, wav2vec_pca = _pca_projection(
        wav2vec_embeddings,
        plot_metadata,
        random_state
    )
    mlp_projection, mlp_pca = _pca_projection(
        mlp_embeddings,
        plot_metadata,
        random_state
    )

    palette = dict(zip(EMOTION_NAMES, sns.color_palette("tab10", len(EMOTION_NAMES))))
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    plot_specs = [
        (wav2vec_projection, wav2vec_pca, "Frozen wav2vec2 mean+std features"),
        (mlp_projection, mlp_pca, "Black-box penultimate representations"),
    ]

    for ax, (projection, pca, title) in zip(axes, plot_specs):
        sns.scatterplot(
            data=projection,
            x="pca_1",
            y="pca_2",
            hue="emotion",
            hue_order=EMOTION_NAMES,
            palette=palette,
            s=28,
            alpha=0.75,
            linewidth=0,
            ax=ax
        )
        # the explained variance is retrived from the eigenvectors of the SVD
        # decomposition -> the bigger the eigenvector the bigger the variance
        # captured by that direction
        explained = pca.explained_variance_ratio_
        ax.set_title(
            f"{title}\nexplained variance: {explained[0]:.1%} + {explained[1]:.1%}"
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.2)

    handles, labels = axes[-1].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.legend(
        handles,
        labels,
        title="Emotion",
        loc="center right",
        bbox_to_anchor=(1.08, 0.5)
    )

    saved_output_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_output_path = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure": fig,
        "axes": axes,
        "output_path": saved_output_path,
        "wav2vec_projection": wav2vec_projection,
        "wav2vec_pca": wav2vec_pca,
        "mlp_projection": mlp_projection,
        "mlp_pca": mlp_pca
    }
