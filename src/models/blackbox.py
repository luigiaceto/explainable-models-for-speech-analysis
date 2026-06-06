from __future__ import annotations
import torch
from torch import nn


class BlackBoxEmotionClassifier(nn.Module):
    """MLP baseline trained on frozen mean+std pooled audio embeddings.

    audio
    -> frozen audio encoder
    -> pooled embedding
    -> Linear input_dim -> LAYER_DIMS[0]
    -> GELU
    -> ...
    -> Linear LAYER_DIMS[n-2] -> LAYER_DIMS[n-1]
    -> GELU
    -> Linear LAYER_DIMS[n-1] -> 4 emotions
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: tuple[int, int] = (256, 128),
        num_classes: int = 4,
        dropout: float = 0.2,
        activation: str = "gelu"
    ) -> None:
        super().__init__()
        activation_layer = _activation_layer(activation)

        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]

        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    activation_layer(),
                    nn.Dropout(dropout)
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _activation_layer(name: str) -> type[nn.Module]:
    normalized_name = name.lower()
    if normalized_name == "gelu":
        return nn.GELU
    if normalized_name == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported activation: {name}")
