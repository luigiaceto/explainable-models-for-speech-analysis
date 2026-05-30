from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.data.crema_d import EMOTION_NAMES as CREMA_D_EMOTION_NAMES
from src.data.meld import EMOTION_NAMES as MELD_EMOTION_NAMES
from src.data.common import load_features, make_feature_loader
from src.data.split import SAMPLE_STRATIFIED_SPLIT, create_splits
from src.evaluation.metrics import compute_summary_classification_metrics
from src.models.blackbox import BlackBoxEmotionClassifier
from src.utils.utils import device_or_default, set_seed


@dataclass
class TrainingConfig:
    input_dim: int = 1536 # MLP input
    feature_extractor_name: str | None = None
    encoder_embedding_dim: int | None = None
    pooling: str | None = None
    hidden_dims: tuple[int, int] = (256, 128) # MLP progressive projection dims
    # num_classes: int = 6
    dropout: float = 0.2
    activation: str = "gelu"
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    split_strategy: str = SAMPLE_STRATIFIED_SPLIT
    # speaker_column: str = "actor_id"
    random_state: int = 42
    num_workers: int = 0
    use_class_weights: bool = True
    early_stopping_patience: int = 10 # if the model doesn't improve after N epochs we stop the training
    monitor_metric: str = "macro_f1" # metric used to check if the model is improving on the validation set
    lr_scheduler: str | None = "reduce_on_plateau"
    scheduler_monitor_metric: str = "macro_f1"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    device: str | None = None
    verbose: bool = True
    dataset_name: str | None = "crema_d"


def _classification_loss(
    metadata: pd.DataFrame,
    config: TrainingConfig,
    device: torch.device,
    num_classes: int = 6,
) -> nn.Module:
    if not config.use_class_weights:
        return nn.CrossEntropyLoss()

    train_labels = metadata.loc[metadata["split"] == "train", "label"].to_numpy()
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_labels
    )
    return nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


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
    metrics = compute_summary_classification_metrics(y_true_array, y_pred_array)
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

    if config.dataset_name == "crema_d":
        num_classes = 6
        speaker_column = "actor_id"
        emotion_names = CREMA_D_EMOTION_NAMES
    elif config.dataset_name == "meld":
        num_classes = 7
        speaker_column = "Speaker"
        emotion_names = MELD_EMOTION_NAMES
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")

    features, metadata = load_features(feature_dir)
    if features.shape[1] != config.input_dim:
        raise ValueError(
            f"Expected feature dim {config.input_dim}, got {features.shape[1]}"
        )
    if config.dataset_name == "crema_d":
        metadata = create_splits(
            metadata=metadata,
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            random_state=config.random_state,
            split_strategy=config.split_strategy,
            speaker_column=speaker_column
        )
    splits_path = output_dir / "splits.csv"
    metadata.to_csv(splits_path, index=False)

    device = device_or_default(config.device)
    train_loader = make_feature_loader(
        features,
        metadata,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split_name="train",
        shuffle=True
    )
    val_loader = make_feature_loader(
        features,
        metadata,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split_name="val",
        shuffle=False
    )

    model = BlackBoxEmotionClassifier(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=num_classes,
        dropout=config.dropout,
        activation=config.activation
    ).to(device)
    
    criterion = _classification_loss(metadata, config, device, num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = None
    if config.lr_scheduler is not None:
        if config.lr_scheduler != "reduce_on_plateau":
            raise ValueError("Only 'reduce_on_plateau' is supported as lr_scheduler")
        scheduler_mode = "min" if config.scheduler_monitor_metric == "loss" else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )

    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    best_train_metrics: dict[str, float] | None = None
    best_val_metrics: dict[str, float] | None = None
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in tqdm(range(1, config.epochs + 1), desc="Training black-box model"):
        epoch_learning_rate = _current_learning_rate(optimizer)
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        if config.scheduler_monitor_metric not in val_metrics:
            raise ValueError(
                f"Unsupported scheduler_monitor_metric '{config.scheduler_monitor_metric}'. "
                f"Available metrics are: {sorted(val_metrics)}"
            )
        if config.monitor_metric not in val_metrics:
            raise ValueError(
                f"Unsupported monitor_metric '{config.monitor_metric}'. "
                f"Available metrics are: {sorted(val_metrics)}"
            )
        score = float(val_metrics[config.monitor_metric])
        scheduler_score = float(val_metrics[config.scheduler_monitor_metric])

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": epoch_learning_rate
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
                    "label_names": emotion_names,
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

        if scheduler is not None:
            scheduler.step(scheduler_score)

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
                f"lr {epoch_learning_rate:.2e} | "
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
