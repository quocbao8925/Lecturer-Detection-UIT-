import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
#               OSNet Building Blocks
# -------------------------------------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16, num_gates=None):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, num_gates)

    def forward(self, x):
        N, C, H, W = x.size()
        y = x.mean(dim=(2,3))
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        return torch.sigmoid(y).view(N, -1, 1, 1)


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = out_channels // reduction

        self.conv1 = ConvLayer(in_channels, mid_channels, 1)

        self.conv2a = ConvLayer(mid_channels, mid_channels, 3, padding=1)
        self.conv2b = ConvLayer(mid_channels, mid_channels, 3, padding=1)
        self.conv2c = ConvLayer(mid_channels, mid_channels, 3, padding=1)
        self.conv2d = ConvLayer(mid_channels, mid_channels, 3, padding=1)

        self.gate = ChannelGate(out_channels)

        self.conv3 = ConvLayer(mid_channels * 4, out_channels, 1)

        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x2a)
        x2c = self.conv2c(x2b)
        x2d = self.conv2d(x2c)

        x_concat = torch.cat([x2a, x2b, x2c, x2d], dim=1)
        x3 = self.conv3(x_concat)

        g = self.gate(x3)
        out = x3 * g

        if self.downsample is not None:
            identity = self.downsample(identity)

        return out + identity


# -------------------------------------------------------
#               OSNet x0.25 (full model)
# -------------------------------------------------------

class OSNet_x025(nn.Module):
    def __init__(self, num_features=512):
        super().__init__()

        self.conv1 = ConvLayer(3, 16, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.block1 = OSBlock(16, 32)
        self.block2 = OSBlock(32, 64)
        self.block3 = OSBlock(64, 128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# -------------------------------------------------------
#               EXPORT TO ONNX
# -------------------------------------------------------

if __name__ == "__main__":
    model = OSNet_x025()
    model.eval()

    dummy = torch.randn(1, 3, 256, 128)

    torch.onnx.export(
        model,
        dummy,
        "osnet_x0_25.onnx",
        input_names=["input"],
        output_names=["embedding"],
        opset_version=11,
        do_constant_folding=True
    )

    print("✔ DONE! Exported osnet_x0_25.onnx (batch=1).")