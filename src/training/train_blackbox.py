from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.data.crema_d import CremaDFeatureDataset, EMOTION_NAMES, load_features
from src.evaluation.metrics import compute_classification_metrics
from src.models.blackbox import BlackBoxEmotionClassifier
from src.utils.utils import device_or_default, set_seed


@dataclass
class TrainingConfig:
    input_dim: int = 1536 # MLP input
    hidden_dims: tuple[int, int] = (256, 128) # MLP progressive projection dims
    num_classes: int = 6
    dropout: float = 0.2
    activation: str = "gelu"
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    num_workers: int = 0
    use_class_weights: bool = True
    early_stopping_patience: int = 10 # if the model doesn't improve after N epochs we stop the training
    monitor_metric: str = "macro_f1" # metric used to check if the model is improving on the validation set
    device: str | None = None
    verbose: bool = True


def create_stratified_splits(
    metadata: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int
) -> pd.DataFrame:
    """Create stratified train/validation/test splits by emotion label."""
    metadata = metadata.reset_index(drop=True)
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total:.4f}")

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
    return splits


def _make_loader(
    features: np.ndarray,
    metadata: pd.DataFrame,
    split_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool
) -> DataLoader:
    indices = metadata.index[metadata["split"] == split_name].tolist()

    dataset = CremaDFeatureDataset(
        features=features,
        metadata=metadata,
        indices=indices
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def _classification_loss(
    metadata: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device
) -> nn.Module:
    if not config.use_class_weights:
        return nn.CrossEntropyLoss()

    train_labels = metadata.loc[metadata["split"] == "train", "label"].to_numpy()
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(config.num_classes),
        y=train_labels
    )
    return nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    y_true = []
    y_pred = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true_array = np.concatenate(y_true)
    y_pred_array = np.concatenate(y_pred)
    metrics = compute_classification_metrics(y_true_array, y_pred_array, EMOTION_NAMES)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_blackbox(
    feature_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None
) -> dict[str, Any]:
    """Train the black-box baseline on precomputed features."""
    # if a TrainingConfig object isn't passed, we use the default one 
    config = config or TrainingConfig()
    set_seed(config.random_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, metadata = load_features(feature_dir)
    if features.shape[1] != config.input_dim:
        raise ValueError(
            f"Expected feature dim {config.input_dim}, got {features.shape[1]}"
        )

    metadata = create_stratified_splits(
        metadata=metadata,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state
    )
    splits_path = output_dir / "splits.csv"
    metadata.to_csv(splits_path, index=False)

    device = device_or_default(config.device)
    train_loader = _make_loader(
        features,
        metadata,
        config.batch_size,
        config.num_workers,
        split_name="train",
        shuffle=True
    )
    val_loader = _make_loader(
        features,
        metadata,
        config.batch_size,
        config.num_workers,
        split_name="val",
        shuffle=False
    )

    model = BlackBoxEmotionClassifier(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=config.num_classes,
        dropout=config.dropout,
        activation=config.activation
    ).to(device)
    
    criterion = _classification_loss(metadata, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    best_train_metrics: dict[str, float] | None = None
    best_val_metrics: dict[str, float] | None = None
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in tqdm(range(1, config.epochs + 1), desc="Training black-box model"):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        score = float(val_metrics[config.monitor_metric])

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"]
        }
        history.append(history_row)

        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            best_train_metrics = {
                "loss": float(train_metrics["loss"]),
                "accuracy": float(train_metrics["accuracy"]),
                "macro_f1": float(train_metrics["macro_f1"]),
                "weighted_f1": float(train_metrics["weighted_f1"])
            }
            best_val_metrics = {
                "loss": float(val_metrics["loss"]),
                "accuracy": float(val_metrics["accuracy"]),
                "macro_f1": float(val_metrics["macro_f1"]),
                "weighted_f1": float(val_metrics["weighted_f1"])
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "label_names": EMOTION_NAMES,
                    "best_epoch": best_epoch,
                    "best_val_score": best_score,
                    "best_train_metrics": best_train_metrics,
                    "best_val_metrics": best_val_metrics,
                    "splits_path": str(splits_path)
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if config.verbose:
            status = "best" if improved else f"patience {epochs_without_improvement}/{config.early_stopping_patience}"
            tqdm.write(
                "Epoch "
                f"{epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f}, "
                f"acc {train_metrics['accuracy']:.4f}, "
                f"macro F1 {train_metrics['macro_f1']:.4f} | "
                f"val loss {val_metrics['loss']:.4f}, "
                f"acc {val_metrics['accuracy']:.4f}, "
                f"macro F1 {val_metrics['macro_f1']:.4f}, "
                f"weighted F1 {val_metrics['weighted_f1']:.4f} | "
                f"{status}"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8"
    )

    if best_train_metrics is None or best_val_metrics is None:
        raise RuntimeError("Training finished without saving a best checkpoint.")

    print(
        "\nBest checkpoint summary\n"
        f"  Epoch:      {best_epoch}\n"
        f"  Train:      accuracy {best_train_metrics['accuracy']:.4f}, "
        f"macro F1 {best_train_metrics['macro_f1']:.4f}, "
        f"weighted F1 {best_train_metrics['weighted_f1']:.4f}\n"
        f"  Validation: accuracy {best_val_metrics['accuracy']:.4f}, "
        f"macro F1 {best_val_metrics['macro_f1']:.4f}, "
        f"weighted F1 {best_val_metrics['weighted_f1']:.4f}"
    )

    return {
        "checkpoint": checkpoint_path,
        "splits": splits_path,
        "history": history_path,
        "best_epoch": best_epoch,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics
    }
