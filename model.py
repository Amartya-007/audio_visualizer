import torch
import torch.nn as nn
from typing import Optional, Union


class ResidualBlock(nn.Module):
    """
    Implements a basic Residual Block for ResNet-like architectures.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for convolution layers (default = 1)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor, fmap_dict: Optional[dict[str, torch.Tensor]] = None, prefix: str = "") -> torch.Tensor:
        """
        Forward pass of the Residual Block.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            fmap_dict (Optional[dict]): Dictionary to store intermediate feature maps
            prefix (str): Prefix key for the feature maps

        Returns:
            Tensor: Output tensor after residual addition and ReLU
        """
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out


class AudioCNN(nn.Module):
    """
    A deep Convolutional Neural Network for audio classification based on ResNet.

    Args:
        num_classes (int): Number of output classes for classification
    """

    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-style block groups
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for _ in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128,
                           stride=2 if i == 0 else 1) for i in range(4)]
        )
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256,
                           stride=2 if i == 0 else 1) for i in range(6)]
        )
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512,
                           stride=2 if i == 0 else 1) for i in range(3)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_feature_maps: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        """
        Forward pass through the AudioCNN.

        Args:
            x (Tensor): Input tensor of shape (B, 1, H, W)
            return_feature_maps (bool): If True, return intermediate feature maps for visualization

        Returns:
            Tensor or (Tensor, dict): Final logits or (logits, feature_maps)
        """
        if not return_feature_maps:
            x = self.conv1(x)
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

        # Feature map tracking mode
        feature_maps: dict[str, torch.Tensor] = {}
        x = self.conv1(x)
        feature_maps["conv1"] = x

        for i, block in enumerate(self.layer1):
            x = block(x, feature_maps, prefix=f"layer1.block{i}")
        feature_maps["layer1"] = x

        for i, block in enumerate(self.layer2):
            x = block(x, feature_maps, prefix=f"layer2.block{i}")
        feature_maps["layer2"] = x

        for i, block in enumerate(self.layer3):
            x = block(x, feature_maps, prefix=f"layer3.block{i}")
        feature_maps["layer3"] = x

        for i, block in enumerate(self.layer4):
            x = block(x, feature_maps, prefix=f"layer4.block{i}")
        feature_maps["layer4"] = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, feature_maps
