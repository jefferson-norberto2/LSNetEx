from torch import mean, max, cat
from torch.nn import Module, AdaptiveAvgPool2d, AdaptiveMaxPool2d, ReLU, Sequential, Sigmoid, Conv2d


class ChannelAttention(Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d(1)

        self.fc = Sequential(
            Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            ReLU(),
            Conv2d(in_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        avg_out = mean(x, dim=1, keepdim=True)
        max_out, _ = max(x, dim=1, keepdim=True)
        x = cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out
