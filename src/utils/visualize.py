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
from src.models.prototype_clustering import load_prototype_clustering_classifier


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


def _plot_indices_for_split(
    metadata: pd.DataFrame,
    split: str,
    sample_limit: int | None,
    random_state: int
) -> np.ndarray:
    if split == "all":
        plot_indices = np.arange(len(metadata))
    else:
        plot_indices = metadata.index[metadata["split"] == split].to_numpy()

    if len(plot_indices) == 0:
        raise ValueError(f"No samples found for split '{split}'.")

    if sample_limit is not None and len(plot_indices) > sample_limit:
        rng = np.random.default_rng(random_state)
        plot_indices = np.sort(rng.choice(plot_indices, size=sample_limit, replace=False))

    return plot_indices


def _finish_figure(
    fig: plt.Figure,
    output_path: str | Path | None,
    dpi: int,
    show: bool
) -> Path | None:
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

    return saved_output_path


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
    """Plot PCA projections of frozen features and black-box representations."""
    features, feature_metadata = load_features(feature_dir, mmap_mode="r")
    split_metadata = pd.read_csv(splits_csv)

    if feature_metadata["file_name"].tolist() != split_metadata["file_name"].tolist():
        raise ValueError("Feature metadata and split metadata are not aligned.")

    plot_indices = _plot_indices_for_split(
        metadata=split_metadata,
        split=split,
        sample_limit=sample_limit,
        random_state=random_state
    )

    plot_metadata = split_metadata.iloc[plot_indices].reset_index(drop=True)
    feature_embeddings = np.asarray(features[plot_indices], dtype=np.float32)
    mlp_embeddings = _blackbox_penultimate_embeddings(
        embeddings=feature_embeddings,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device
    )

    # WavLM pooled embeddings
    feature_projection, feature_pca = _pca_projection(
        feature_embeddings,
        plot_metadata,
        random_state
    )

    # black-box penultimate embeddings
    mlp_projection, mlp_pca = _pca_projection(
        mlp_embeddings,
        plot_metadata,
        random_state
    )

    palette = dict(zip(EMOTION_NAMES, sns.color_palette("tab10", len(EMOTION_NAMES))))
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    split_label = "all splits" if split == "all" else f"{split} split"
    plot_specs = [
        (feature_projection, feature_pca, "Frozen audio encoder mean+std features"),
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
        explained = pca.explained_variance_ratio_
        ax.set_title(
            f"{title}\n{split_label} | explained variance: "
            f"{explained[0]:.1%} + {explained[1]:.1%}"
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

    saved_output_path = _finish_figure(fig, output_path, dpi, show)

    return {
        "figure": fig,
        "axes": axes,
        "output_path": saved_output_path,
        "feature_projection": feature_projection,
        "feature_pca": feature_pca,
        "mlp_projection": mlp_projection,
        "mlp_pca": mlp_pca
    }


def plot_prototype_embedding_pca(
    embedding_dir: str | Path,
    model_dir: str | Path,
    split: str = "all",
    output_path: str | Path | None = None,
    sample_limit: int | None = None,
    random_state: int = 42,
    show: bool = True,
    figsize: tuple[float, float] = (9, 7),
    dpi: int = 180
) -> dict[str, Any]:
    """Plot black-box embeddings and overlay real training prototypes.

    The prototypes are the saved real training samples in ``prototypes.npy``.
    They are projected together with the selected dataset embeddings so their
    position is shown in the same PCA coordinate system.
    """
    embeddings, metadata = load_features(embedding_dir, mmap_mode="r")
    classifier, _ = load_prototype_clustering_classifier(model_dir)

    plot_indices = _plot_indices_for_split(
        metadata=metadata,
        split=split,
        sample_limit=sample_limit,
        random_state=random_state
    )
    plot_metadata = metadata.iloc[plot_indices].reset_index(drop=True)
    plot_embeddings = np.asarray(embeddings[plot_indices], dtype=np.float32)
    prototypes = np.asarray(classifier.prototypes, dtype=np.float32)
    prototype_labels = np.asarray(classifier.prototype_labels, dtype=np.int64)

    combined_embeddings = np.vstack([plot_embeddings, prototypes])
    scaled_embeddings = StandardScaler().fit_transform(combined_embeddings)
    pca = PCA(n_components=2, random_state=random_state)
    coordinates = pca.fit_transform(scaled_embeddings)
    sample_coordinates = coordinates[: len(plot_embeddings)]
    prototype_coordinates = coordinates[len(plot_embeddings) :]

    projection = plot_metadata[["file_name", "emotion", "label", "split"]].copy()
    projection["pca_1"] = sample_coordinates[:, 0]
    projection["pca_2"] = sample_coordinates[:, 1]

    prototype_projection = pd.DataFrame(
        {
            "prototype_position": np.arange(len(prototypes)),
            "label": prototype_labels,
            "emotion": [EMOTION_NAMES[label] for label in prototype_labels],
            "pca_1": prototype_coordinates[:, 0],
            "pca_2": prototype_coordinates[:, 1]
        }
    )

    prototype_metadata_path = Path(model_dir) / "prototype_metadata.csv"
    if prototype_metadata_path.exists():
        prototype_metadata = pd.read_csv(prototype_metadata_path)
        prototype_projection = prototype_projection.merge(
            prototype_metadata,
            on="prototype_position",
            how="left",
            suffixes=("", "_source")
        )

    palette = dict(zip(EMOTION_NAMES, sns.color_palette("tab10", len(EMOTION_NAMES))))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    sns.scatterplot(
        data=projection,
        x="pca_1",
        y="pca_2",
        hue="emotion",
        hue_order=EMOTION_NAMES,
        palette=palette,
        s=24,
        alpha=0.35,
        linewidth=0,
        ax=ax
    )

    for emotion in EMOTION_NAMES:
        emotion_prototypes = prototype_projection[
            prototype_projection["emotion"] == emotion
        ]
        if emotion_prototypes.empty:
            continue
        ax.scatter(
            emotion_prototypes["pca_1"],
            emotion_prototypes["pca_2"],
            marker="X",
            s=170,
            c=[palette[emotion]],
            edgecolors="black",
            linewidths=1.2,
            label=f"{emotion} prototype"
        )

    explained = pca.explained_variance_ratio_
    split_label = "all splits" if split == "all" else f"{split} split"
    ax.set_title(
        "Prototype clustering\n"
        f"{split_label} | explained variance: {explained[0]:.1%} + {explained[1]:.1%}"
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    emotion_handles = handles[: len(EMOTION_NAMES)]
    emotion_labels = labels[: len(EMOTION_NAMES)]
    ax.legend(
        emotion_handles,
        emotion_labels,
        title="Emotion",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    saved_output_path = _finish_figure(fig, output_path, dpi, show)

    return {
        "figure": fig,
        "axes": ax,
        "output_path": saved_output_path,
        "projection": projection,
        "prototype_projection": prototype_projection,
        "pca": pca
    }
