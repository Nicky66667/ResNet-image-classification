import torch.nn as nn
import torch

# Basic block used in ResNet architecture
class BasicBlock(nn.Module):
    # Factor by which the output channels are increased (1 in this case)
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        Basic Block for ResNet. It contains two convolutional layers with skip connections.

        Parameters:
        - in_channel: Number of input channels (e.g., for RGB image, 3)
        - out_channel: Number of output channels after block
        - stride: Stride for convolution, default is 1
        - downsample: Used to match input/output dimensions if needed
        """

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel) # Batch normalization after convolution
        self.relu = nn.ReLU()   # ReLU activation
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel) # Batch normalization
        self.downsample = downsample # For matching input/output dimensions if needed

    def forward(self, x):
        """
            Forward pass through the block.

            Parameters:
            - x: Input tensor

            Returns:
            - Output tensor after processing through the basic block
        """

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # Skip connection (residual connection)
        out = self.relu(out)

        return out

# Bottleneck block used in deeper ResNet architectures (ResNet50, ResNet101)
class Bottleneck(nn.Module):
    """
    Reference: Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4 # Output channels are multiplied by 4 for this block

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        """
            Parameters:
            - in_channel: Number of input channels
            - out_channel: Number of output channels
            - stride: Stride for convolutions
            - downsample: Used to match input/output dimensions if necessary
            - groups: Defines group convolution (for ResNext architectures)
            - width_per_group: Defines the width for the grouped convolution
        """

        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        # First 1x1 convolution to reduce the number of channels (squeeze)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)

        # # Second 3x3 convolution with groups (for group convolution in ResNext)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        # Third 1x1 convolution to restore the channels (unsqueeze)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True) # ReLU activation
        self.downsample = downsample # For matching input/output dimensions if needed

    def forward(self, x):
        """
            Forward pass through the bottleneck block.

            Parameters:
            - x: Input tensor

            Returns:
            - Output tensor after processing through the bottleneck block
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity # Skip connection (residual connection)
        out = self.relu(out)

        return out

# ResNet model class
class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        """
            Parameters:
            - block: Block type (either BasicBlock or Bottleneck)
            - blocks_num: List of integers defining the number of blocks in each layer
            - num_classes: Number of output classes (e.g., 1000 for ImageNet)
            - include_top: If True, include the fully connected layer at the top
            - groups: Defines the number of groups for grouped convolutions (used in ResNext)
            - width_per_group: Defines width for grouped convolutions
        """

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 # Initial input channel size

        self.groups = groups
        self.width_per_group = width_per_group

        # Initial convolution and pooling layers
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Creating each layer of the ResNet
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # Final fully connected layer (if include_top is True)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize the weights using Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
            Helper function to create a layer with multiple blocks.

            Parameters:
            - block: The block type (BasicBlock or Bottleneck)
            - channel: Number of output channels for the blocks in this layer
            - block_num: Number of blocks in this layer
            - stride: Stride for the first block (used for downsampling)

            Returns:
            - A sequential container of the blocks
        """

        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
            Forward pass through the entire ResNet model.

            Parameters:
            - x: Input tensor

            Returns:
            - Output tensor after processing through all layers
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

# Factory functions to create different versions of ResNet
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
