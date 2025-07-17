import torch
import torch.nn as nn


# * ResidualBlock implements a basic residual block for CNNs, allowing shortcut connections for better gradient flow.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # * First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # * Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # * Shortcut connection for matching dimensions or stride
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            # ! The shortcut is essential when dimensions or stride change, to allow for element-wise addition.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, fmap_dict=None, prefix=""):
        # * Forward pass through the residual block
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # * Apply shortcut connection
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut

        # ? Should we store feature maps before or after the final ReLU?
        # * Optionally store feature maps for visualization or debugging
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out


# * AudioCNN is a deep convolutional neural network for audio classification, based on the ResNet architecture.
class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # * Initial convolutional block for low-level feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))

        # * Four main layers composed of stacked residual blocks
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])

        # * Final layers for classification
        # * Global average pooling to create a fixed-size feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)  # * Dropout for regularization
        # * Fully connected layer for final classification
        
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        # * if we dont want to see feature maps, we just do a standard forward pass
        if not return_feature_maps:
            # * Standard forward pass for training and inference
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
            x = x.view(x.size(0), -1)  # * Flatten the features
            x = self.dropout(x)
            x = self.fc(x)
            return x
        else:
            #! THIS WILL SHOW THE FEATURE MAPS
            # * if want to see feature maps, we store them in a dictionary 
            feature_maps = {}
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
