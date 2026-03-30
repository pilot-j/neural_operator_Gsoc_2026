import torch.nn as nn


class LiftingLayer(nn.Module):
    """Projects input to latent space via 2-layer MLP with expansion.

    Modern neuraloperator uses 2x expansion. For in_channels=3 (1 + 2 grid),
    this goes: 3 -> 2*hidden -> hidden.
    """

    def __init__(self, in_channels, hidden_channels, expansion=2):
        super().__init__()
        mid = int(hidden_channels * expansion)
        self.fc1 = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.activation = nn.GELU()
        self.fc2 = nn.Conv2d(mid, hidden_channels, kernel_size=1)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class NaiveLiftingLayer(nn.Module):
    """Projects input from physical space to higher-dimensional latent space."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        return self.fc(x)
