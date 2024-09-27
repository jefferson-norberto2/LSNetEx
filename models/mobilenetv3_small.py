from torch import Tensor
from torch.nn import Module
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3Small(Module):
    """
    Extended version of MobileNetV3 model.

    Args:
        pretrained (bool): Use pretrained network.
    """
    def __init__(self, pretrained=True) -> None:
        super(MobileNetV3Small, self).__init__()
        _weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        _model = mobilenet_v3_small(weights = _weights)
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

        x = self.features[0:1](x)
        out1 = x
        x = self.features[1:2](x)
        out2 = x
        x = self.features[2:4](x)
        out3 = x
        x = self.features[4:6](x)
        out4 = x
        x = self.features[6:13](x)
        out5 = x
        
        return out1, out2, out3, out4, out5
    

if __name__ == '__main__':
    import torch
    model = MobileNetV3Small(pretrained=True)
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
