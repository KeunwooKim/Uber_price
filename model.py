from typing import Optional, Callable
import torch.nn as nn
from torch import Tensor


def batchnorm_layer(
        bn: Optional[Callable[..., nn.Module]] = None,
        features: int = 0
) -> nn.Module:
    return bn(features) if bn else nn.BatchNorm1d(features)


class DoubleLinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_norm: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0
    ) -> None:
        super(DoubleLinearBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            batchnorm_layer(batch_norm, out_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_features, out_features),
            batchnorm_layer(batch_norm, out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Model(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: list,
        out_features: int,
        batch_norm: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
        init_weights: bool = True
    ) -> None:
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(DoubleLinearBlock(in_features, hidden_features[0], batch_norm, dropout))

        for i in range(len(hidden_features) - 1):
            self.layers.append(DoubleLinearBlock(hidden_features[i], hidden_features[i + 1], batch_norm, dropout))

        self.layers.append(nn.Linear(hidden_features[-1], out_features))

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

