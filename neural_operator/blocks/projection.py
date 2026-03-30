import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    """Projects from latent space to class logits via 1x1 convs + global pooling."""

    def __init__(self, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.norm = nn.InstanceNorm2d(hidden_channels // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.gelu(self.norm(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x


class NaiveProjectionLayer(nn.Module):
    """Projects from latent space to class logits via 1x1 convs + global pooling."""

    def __init__(self, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1)
        self.norm = nn.InstanceNorm2d(hidden_channels // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.gelu(self.norm(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
