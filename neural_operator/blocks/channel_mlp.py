import torch.nn as nn


class ChannelMLP(nn.Module):
    """2-layer channel mixing MLP (analogous to FFN in Transformers).

    Applied pointwise (1x1 convs) with bottleneck expansion.
    Modern neuraloperator library uses 0.5x expansion (bottleneck).
    """

    def __init__(self, channels, expansion=0.5, dropout=0.0):
        super().__init__()
        hidden = int(channels * expansion)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
