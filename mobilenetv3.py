import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import SqueezeExcitation, ConvBNActivation # InvertedResidual, InvertedResidualConfig

__all__ = ['MobileNetV3', 'mobilenet_v3']


model_urls = {
    'mobilenet_v3': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}


class InvertedResidualConfig:
    def __init__(self, input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride):
        self.input_channels = input_channels
        self.kernel = kernel
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.use_se = use_se
        self.activation = activation
        self.stride = stride


class InvertedResidual(nn.Module):
    def __init__(self, config):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels
        layers = []
        if config.expanded_channels != config.input_channels:
            layers.append(ConvBNActivation(config.input_channels, config.expanded_channels, kernel_size=1, stride=1))
        layers.extend([
            ConvBNActivation(config.expanded_channels, config.expanded_channels, kernel_size=config.kernel, stride=config.stride, groups=config.expanded_channels),
            SqueezeExcitation(config.expanded_channels, config.expanded_channels // 4) if config.use_se else nn.Identity(),
            nn.Conv2d(config.expanded_channels, config.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(config.out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs

        input_channels = 16
        self.last_channels = 320

        features = [ConvBNActivation(3, input_channels, kernel_size=3, stride=2)]
        
        for cfg in cfgs:
            cfg.input_channels = int(cfg.input_channels * width_mult)
            cfg.expanded_channels = int(cfg.expanded_channels * width_mult)
            cfg.out_channels = int(cfg.out_channels * width_mult)
            features.append(InvertedResidual(cfg))
            
        features.append(ConvBNActivation(cfgs[-1].out_channels, self.last_channels, kernel_size=1, stride=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(self.last_channels, 320, kernel_size=1, stride=1, padding=0),
                            nn.Hardswish(inplace=True),
                            nn.Conv2d(320, num_classes, kernel_size=1, stride=1, padding=0),
                        )

        
    def _forward_impl(self, x):
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


    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v3_small(pretrained=True, **kwargs):
    cfgs = [
        InvertedResidualConfig(16, 3, 16, 16, True, nn.ReLU, 2),
        InvertedResidualConfig(16, 3, 72, 24, False, nn.ReLU, 2),
        InvertedResidualConfig(24, 3, 88, 24, False, nn.ReLU, 1),
        InvertedResidualConfig(24, 5, 96, 40, True, nn.Hardswish, 2),
        InvertedResidualConfig(40, 5, 240, 40, True, nn.Hardswish, 1),
        InvertedResidualConfig(40, 5, 240, 40, True, nn.Hardswish, 1),
        InvertedResidualConfig(40, 3, 120, 48, True, nn.Hardswish, 1),
        InvertedResidualConfig(48, 3, 144, 48, True, nn.Hardswish, 1),
        InvertedResidualConfig(48, 5, 288, 96, True, nn.Hardswish, 2),
        InvertedResidualConfig(96, 5, 576, 96, True, nn.Hardswish, 1),
        InvertedResidualConfig(96, 5, 576, 96, True, nn.Hardswish, 1),
    ]
    return MobileNetV3(cfgs)

if __name__=='__main__':
    import torch
    model = mobilenet_v3_small()
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)