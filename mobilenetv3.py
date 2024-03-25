import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['MobileNetV3_Small', 'mobilenet_v3_small']

model_urls = {
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_1x1_bn(inp, oup, relu6=True):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)
    )

def conv_3x3_bn(inp, oup, stride, relu6=True):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)
    )

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        
        squeeze_channels = input_channels // squeeze_factor
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_channels, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidualSE(nn.Module):
    def __init__(self, input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation, width_mult):
        super(InvertedResidualSE, self).__init__()

        # Convolução 1x1
        self.conv1 = nn.Conv2d(input_channels, expanded_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.activation = activation()

        # Convolução 3x3
        self.conv2 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel, stride=stride, padding=kernel // 2, dilation=dilation, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)

        # SE (Squeeze-and-Excitation) se usar SE
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(expanded_channels)

        # Convolução 1x1 final
        self.conv3 = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Camada de atalho se houver mudança de tamanho
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Caminho principal
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        if self.use_se:
            out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Caminho de atalho
        shortcut = self.shortcut(x)

        # Soma do caminho principal com o caminho de atalho
        out += shortcut
        out = self.activation(out)

        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV3_Small, self).__init__()

        # Configuration for MobileNetV3 Small
        self.cfgs = [
            # k, t, c, SE, s, activation, stride, dilation, width_mult
            [3, 16,  16, 16, False, nn.ReLU, 2, 1, 1.0],
            [3, 72,  24, 24, False, nn.ReLU, 2, 1, 1.0],
            [3, 88,  24, 24, False, nn.ReLU, 1, 1, 1.0],
            [5, 96,  40, 40, True, nn.Hardswish, 2, 1, 1.0],
            [5, 240, 40, 40, True, nn.Hardswish, 1, 1, 1.0],
            [5, 120, 48, 48, True, nn.Hardswish, 1, 1, 1.0],
            [5, 144, 48, 48, True, nn.Hardswish, 1, 1, 1.0],
            [5, 240, 40, 40, True, nn.Hardswish, 1, 1, 1.0],
            [5, 288, 96, 96, True, nn.Hardswish, 2, 1, 1.0],
            [5, 576, 96, 96, True, nn.Hardswish, 1, 1, 1.0],
            [5, 576, 96, 96, True, nn.Hardswish, 1, 1, 1.0],
        ]

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidualSE
        for k, t, c, SE, s, activation, stride, dilation, width_mult in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(t * width_mult, 8)
            layers.append(block(input_channels=input_channel, kernel=k, expanded_channels=exp_size, out_channels=output_channel, use_se=SE, activation=activation, stride=stride, dilation=dilation, width_mult=width_mult))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(576 * width_mult, 8) if width_mult > 1.0 else 576
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

    def forward(self, x):
        x = self.features[:2](x)
        out1 = x
        x = self.features[2:4](x)
        out2 = x
        x = self.features[4:7](x)
        out3 = x
        x = self.features[7:11](x)
        out4 = x
        x = self.features[11:](x)
        out5 = x

        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)

        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False)

        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=False)

        out4 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=False)

        return out1, out2, out3, out4, out5

def mobilenet_v3_small(pretrained=True, **kwargs):
    """
    Constructs a MobileNetV3 Small architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV3_Small(**kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mobilenet_v3_small'])
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    import torch
    model = mobilenet_v3_small(pretrained=False)
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
