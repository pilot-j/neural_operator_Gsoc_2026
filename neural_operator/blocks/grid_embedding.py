import torch
import torch.nn as nn


class GridEmbedding2d(nn.Module):
    """Appends normalized (x, y) grid coordinates as 2 extra input channels.

    Standard in all FNO implementations — provides position-awareness to
    the otherwise translation-equivariant Fourier layers.
    """

    def forward(self, x):
        B, _, H, W = x.shape
        grid_y = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        grid_x = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, grid_x, grid_y], dim=1)
