import torch.nn as nn

from .spectral_conv import SpectralConv2d, NaiveSpectralConv2d
from .channel_mlp import ChannelMLP


class FNOBlock2d(nn.Module):
    """FNO block: spectral conv + skip -> norm -> activation -> channel MLP + skip -> norm -> activation.

    Dropout modes: 'none', 'spatial', 'spectral', 'both'
    Norm options: 'batch' (default), 'instance', 'group', 'none'
    """

    def __init__(self, channels, modes1, modes2, activation='gelu',
                 dropout=0.1, dropout_mode='both', norm='batch',
                 mlp_expansion=0.5, use_channel_mlp=True):
        super().__init__()
        use_spectral = dropout_mode in ('spectral', 'both')
        use_spatial = dropout_mode in ('spatial', 'both')

        self.spectral_conv = SpectralConv2d(
            channels, channels, modes1, modes2,
            spectral_dropout=dropout if use_spectral else 0.0
        )
        self.linear = nn.Conv2d(channels, channels, kernel_size=1)

        def make_norm(channels):
            if norm == 'batch':
                return nn.BatchNorm2d(channels)
            elif norm == 'instance':
                return nn.InstanceNorm2d(channels)
            elif norm == 'group':
                return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
            else:
                return nn.Identity()

        self.norm1 = make_norm(channels)
        self.norm2 = make_norm(channels)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if use_spatial else nn.Identity()

        # Channel MLP for nonlinear channel mixing
        if use_channel_mlp:
            self.channel_mlp = ChannelMLP(channels, expansion=mlp_expansion, dropout=dropout)
        else:
            self.channel_mlp = None

    def forward(self, x):
        residual = x
        h = self.spectral_conv(x) + self.linear(x)
        h = self.activation(self.norm1(h))
        h = self.dropout(h)
        if self.channel_mlp is not None:
            h = h + self.channel_mlp(h)
            h = self.activation(self.norm2(h))
        return h + residual


class NaiveFNOBlock2d(nn.Module):
    """Single FNO block: spectral conv (global) + 1x1 conv (local) + norm + activation + dropout."""

    def __init__(self, channels, modes1, modes2, activation='gelu', dropout=0.1):
        super().__init__()
        self.spectral_conv = NaiveSpectralConv2d(channels, channels, modes1, modes2)
        self.linear = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(channels)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.norm(self.spectral_conv(x) + self.linear(x))
        x = self.activation(x)
        x = self.dropout(x)
        return x
