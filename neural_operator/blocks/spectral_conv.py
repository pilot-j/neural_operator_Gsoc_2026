import math

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution with ortho-normalized FFT, Kaiming init,
    learnable bias, and optional spectral dropout."""

    def __init__(self, in_channels, out_channels, modes1, modes2, spectral_dropout=0.0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.spectral_dropout = spectral_dropout

        # Kaiming-like init for complex weights (modern neuraloperator convention)
        std = math.sqrt(2.0 / (in_channels + out_channels))
        self.weights1 = nn.Parameter(
            std * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            std * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

        # Learnable bias in spatial domain (after IFFT)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.bias = None

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm='ortho')

        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Spectral dropout: randomly zero out Fourier modes during training
        if self.training and self.spectral_dropout > 0:
            mask = (torch.rand(1, 1, out_ft.size(-2), out_ft.size(-1),
                    device=out_ft.device) > self.spectral_dropout).float()
            out_ft = out_ft * mask

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')

        if self.bias is not None:
            x = x + self.bias

        return x


class NaiveSpectralConv2d(nn.Module):
    """2D Spectral Convolution: learns weights in Fourier space for global receptive field.

    Simpler baseline without ortho-norm FFT, using 1/(in*out) scaling
    and torch.rand initialization.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
