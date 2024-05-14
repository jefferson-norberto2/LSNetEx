from typing import Any, Callable, List
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, MobileNetV3, MobileNet_V3_Small_Weights, InvertedResidualConfig
from torch.nn.functional import interpolate
from typing import Optional
from torchvision.models._utils import _ovewrite_named_param

class MobileNetV3Small(MobileNetV3):
    """
    Extended version of MobileNetV3 model.

    Args:
        inverted_residual_setting (List[InvertedResidualConfig]): List of inverted residual settings.
        last_channel (int): Number of output channels of the final convolutional layer.
        num_classes (int, optional): Number of classes for classification. Defaults to 1000.
        block (Callable[..., Module] | None, optional): Custom block for the architecture. Defaults to None.
        norm_layer (Callable[..., Module] | None, optional): Custom normalization layer. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.2.
        **kwargs (Any): Additional keyword arguments for model initialization.

    """
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, num_classes: int = 1000, block: Callable[..., Module] | None = None, norm_layer: Callable[..., Module] | None = None, dropout: float = 0.2, **kwargs: Any) -> None:
        super().__init__(inverted_residual_setting, last_channel, num_classes, block, norm_layer, dropout, **kwargs)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensors from various intermediate layers.

        """

        x = self.features[:1](x)
        out1 = x
        x = self.features[1:2](x)
        out2 = x
        x = self.features[2:4](x)
        out3 = x
        x = self.features[4:6](x)
        out4 = x
        x = self.features[6:](x)
        out5 = x

        return out1, out2, out3, out4, out5

def mobilenet_v3_small_ex(
    *, weights: Optional[MobileNet_V3_Small_Weights] = MobileNet_V3_Small_Weights.IMAGENET1K_V1, progress: bool = True, **kwargs: Any
) -> MobileNetV3Small:
    """
    Constructs a MobileNetV3 small model with extended functionality.

    Args:
        weights (Optional[MobileNet_V3_Small_Weights], optional): Pre-trained weights. Defaults to None.
        progress (bool, optional): If True, displays a progress bar of the download. Defaults to True.
        **kwargs (Any): Additional keyword arguments for model initialization.

    Returns:
        MobileNetV3Ex: An instance of MobileNetV3Ex model.

    """
    weights = MobileNet_V3_Small_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3Small(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
    

if __name__ == '__main__':
    import torch
    model = mobilenet_v3_small_ex(pretrained=True)
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
