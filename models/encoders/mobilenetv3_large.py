from torch import Tensor
from torch.nn import Module, Conv2d
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3Large(Module):
    """
    Extended version of MobileNetV3 model.

    Args:
        pretrained (bool): pretrained network.
    """
    def __init__(self, pretrained=True) -> None:
        super(MobileNetV3Large, self).__init__()
        _weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        _model = mobilenet_v3_large(weights = _weights)
        self.features = _model.features
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensors from various intermediate layers.

        """
        x = self.features[:2](x)
        out1 = self._transition_layer(16, 32, x)
        x = self.features[2:4](x)
        out2 = self._transition_layer(24, 48, x)
        x = self.features[4:7](x)
        out3 = self._transition_layer(40, 80, x)
        x = self.features[7:10](x)
        out4 = self._transition_layer(80, 160, x)
        x = self.features[10:13](x)
        out5 = self._transition_layer(112, 224, x)
        x = self.features[13:16](x)
        out6 = self._transition_layer(160, 320, x)
        x = self.features[16:](x)
        out7 = x
        

        return out1, out2, out3, out4, out5, out6, out7

    def _transition_layer(self, in_channels, out_channels, feature):
        t = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        t.to('cuda')
        return t(feature)

if __name__ == '__main__':
    import torch
    model = MobileNetV3Large(pretrained=True)
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
