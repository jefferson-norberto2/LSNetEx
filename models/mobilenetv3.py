from typing import Any, Callable, List
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, MobileNetV3, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights, InvertedResidualConfig
from torch.nn.functional import interpolate
from typing import Optional
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param

class MobileNetV3Ex(MobileNetV3):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], last_channel: int, num_classes: int = 1000, block: Callable[..., Module] | None = None, norm_layer: Callable[..., Module] | None = None, dropout: float = 0.2, **kwargs: Any) -> None:
        super().__init__(inverted_residual_setting, last_channel, num_classes, block, norm_layer, dropout, **kwargs)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features[:2](x)
        out1 = x
        x = self.features[2:4](x)
        out2 = x
        x = self.features[4:7](x)
        out3 = x
        x = self.features[7:10](x)
        out4 = x
        x = self.features[10:12](x)
        out5 = x
        x = self.features[12:15](x)
        out6 = x
        x = self.features[15:](x)
        out7 = x
        

        # out1 = interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)

        # out2 = interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False)

        # out3 = interpolate(out3, scale_factor=2, mode='bilinear', align_corners=False)

        out5 = interpolate(out5, scale_factor=0.5, mode='bilinear', align_corners=False)

        return out1, out2, out3, out4, out5, out6, out7

def mobilenet_v3_small_ex(
    *, weights: Optional[MobileNet_V3_Small_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3Ex:
    weights = MobileNet_V3_Small_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return _mobilenet_v3_ex(inverted_residual_setting, last_channel, weights, progress, **kwargs)

def mobilenet_v3_large_ex(
    *, weights: Optional[MobileNet_V3_Large_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV3:
    weights = MobileNet_V3_Large_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3_ex(inverted_residual_setting, last_channel, weights, progress, **kwargs)

def _mobilenet_v3_ex(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MobileNetV3Ex:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3Ex(inverted_residual_setting, last_channel, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

if __name__ == '__main__':
    import torch
    model = mobilenet_v3_large_ex(pretrained=True)
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
