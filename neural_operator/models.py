import torch
import torch.nn as nn

from .blocks import (
    GridEmbedding2d,
    SpectralConv2d,
    NaiveSpectralConv2d,
    FNOBlock2d,
    NaiveFNOBlock2d,
    LiftingLayer,
    NaiveLiftingLayer,
    ProjectionLayer,
    NaiveProjectionLayer,
)


class NaiveFNO(nn.Module):
    """Fourier Neural Operator for Image Classification.

    Naive/original implementation matching the reference FNOImageClassifier:
    - Simple 1/(in*out) weight scaling with torch.rand initialization
    - No ortho-normalized FFT
    - InstanceNorm in blocks
    - No spectral dropout, no channel MLP, no residual connections in blocks
    """

    def __init__(self, in_channels=1, num_classes=3, hidden_channels=64,
                 modes1=12, modes2=12, num_layers=6, activation='gelu', dropout=0.1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.hidden_channels = hidden_channels

        self.lifting = NaiveLiftingLayer(in_channels, hidden_channels)
        self.fno_blocks = nn.ModuleList([
            NaiveFNOBlock2d(hidden_channels, modes1, modes2, activation, dropout)
            for _ in range(num_layers)
        ])
        self.projection = NaiveProjectionLayer(hidden_channels, num_classes, dropout)

    def forward(self, x):
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.projection(x)
        return x


class FNOImageClassifier(nn.Module):
    """Improved FNO for Image Classification.

    Enhancements over NaiveFNO:
    - Grid embedding for position awareness
    - Ortho-normalized FFT with Kaiming initialization
    - Learnable spectral bias
    - Configurable normalization (batch/instance/group)
    - Spectral and spatial dropout modes
    - Channel MLP for nonlinear mixing
    - Residual connections in FNO blocks
    """

    def __init__(self, in_channels=1, num_classes=3, hidden_channels=128,
                 modes=20, num_layers=6, activation='gelu',
                 dropout=0.1, dropout_mode='both', norm='batch',
                 mlp_expansion=0.5, use_channel_mlp=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.num_layers = num_layers
        self.dropout_mode = dropout_mode

        self.grid = GridEmbedding2d()
        self.lifting = LiftingLayer(in_channels + 2, hidden_channels)
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, modes, modes, activation,
                       dropout, dropout_mode, norm,
                       mlp_expansion, use_channel_mlp)
            for _ in range(num_layers)
        ])
        self.projection = ProjectionLayer(hidden_channels, num_classes, dropout)

    def forward(self, x):
        x = self.grid(x)
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.projection(x)
        return x


class HourglassFNO(nn.Module):
    """Hourglass FNO with varying modes, constant channels.

    Architecture (6 layers, base_modes=12):
        Encoder:  12 -> 9 -> 6   (modes decrease)
        Decoder:  6 -> 9 -> 12   (modes increase)

    Channels stay constant throughout.
    Skip connections between encoder and decoder layers.
    """

    def __init__(self, in_channels=1, num_classes=3, hidden_channels=64,
                 base_modes=12, num_layers=6, activation='gelu',
                 dropout=0.1, dropout_mode='both', norm='batch',
                 mlp_expansion=0.5, use_channel_mlp=True,
                 mode_reduction_factor=0.75):
        super().__init__()
        assert num_layers % 2 == 0, "num_layers must be even for symmetric hourglass"

        self.hidden_channels = hidden_channels
        self.base_modes = base_modes
        self.num_layers = num_layers
        self.dropout_mode = dropout_mode

        half_layers = num_layers // 2

        # Modes schedule: decrease in encoder, increase in decoder
        self.encoder_modes = []
        for i in range(half_layers):
            m = max(4, int(base_modes * (mode_reduction_factor ** i)))
            self.encoder_modes.append(m)
        self.decoder_modes = list(reversed(self.encoder_modes))

        # Grid embedding + Lifting
        self.grid = GridEmbedding2d()
        self.lifting = LiftingLayer(in_channels + 2, hidden_channels)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, self.encoder_modes[i], self.encoder_modes[i],
                       activation, dropout, dropout_mode, norm, mlp_expansion, use_channel_mlp)
            for i in range(half_layers)
        ])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, self.decoder_modes[i], self.decoder_modes[i],
                       activation, dropout, dropout_mode, norm, mlp_expansion, use_channel_mlp)
            for i in range(half_layers)
        ])

        # Learnable skip connection scaling
        self.skip_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(half_layers)
        ])

        # Projection head
        self.projection = ProjectionLayer(hidden_channels, num_classes, dropout)

    def forward(self, x):
        x = self.grid(x)
        x = self.lifting(x)

        # Encoder - store outputs for skip connections
        encoder_outputs = []
        for block in self.encoder_blocks:
            encoder_outputs.append(x)
            x = block(x)

        # Decoder with additive skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            # Add scaled skip connection from corresponding encoder layer
            skip_idx = len(encoder_outputs) - i - 1
            if skip_idx >= 0:
                x = x + self.skip_scales[i] * encoder_outputs[skip_idx]

        x = self.projection(x)
        return x
