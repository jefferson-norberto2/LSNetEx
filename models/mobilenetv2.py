from torch import Tensor
from torch.nn import Module
from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2Ex(Module):
    """
    Extended version of MobileNetV2 model from Pytorch.

    """
    def __init__(self, pretrained=True) -> None:
        super(MobileNetV2Ex, self).__init__()
        weigths = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        _model = mobilenet_v2(weights=weigths)
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
        x = self.features[2:3](x)
        out2 = x
        x = self.features[3:7](x)
        out3 = x
        x = self.features[7:14](x)
        out4 = x
        x = self.features[14:18](x)
        out5 = x

        return out1, out2, out3, out4, out5

if __name__ == '__main__':
    import torch
    model = MobileNetV2Ex()
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
