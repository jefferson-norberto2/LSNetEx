from torch import Tensor, cuda
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
        out1 = x
        x = self.features[2:4](x)
        out2 = x
        x = self.features[4:7](x)
        out3 = x
        x = self.features[7:10](x)
        out4 = x
        x = self.features[10:](x)
        out5 = x

        return out1, out2, out3, out4, out5
        
    def _transition_layer(self, in_channels, out_channels, feature):
        transfer = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        if (cuda.is_available()):
            transfer.to('cuda')
        return transfer(feature)

if __name__ == '__main__':
    import torch
    model = MobileNetV3Large(pretrained=True)
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
