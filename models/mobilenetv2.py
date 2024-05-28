from typing import Any, Callable, List
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models.mobilenetv2 import MobileNetV2, MobileNet_V2_Weights
from typing import Optional
from torchvision.models._utils import _ovewrite_named_param

class MobileNetV2Ex(MobileNetV2):
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
    def __init__(self, 
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., Module]] = None,
        norm_layer: Optional[Callable[..., Module]] = None,
        dropout: float = 0.2,
        ) -> None:

        super().__init__(
        num_classes,
        width_mult,
        inverted_residual_setting,
        round_nearest,
        block,
        norm_layer,
        dropout,
        )
    
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


def mobilenet_v2(
    weights: Optional[MobileNet_V2_Weights] = MobileNet_V2_Weights.IMAGENET1K_V2, 
    progress: bool = True, 
    **kwargs: Any,
    ) -> MobileNetV2Ex:
    """MobileNetV2 architecture from the `MobileNetV2: Inverted Residuals and Linear
    Bottlenecks <https://arxiv.org/abs/1801.04381>`_ paper.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.mobilenetv2.MobileNetV2``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V2_Weights
        :members:
    """
    model = MobileNetV2Ex(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

if __name__ == '__main__':
    import torch
    model = mobilenet_v2()
    rgb = torch.randn(1, 3, 224, 224)
    out = model(rgb)
    for i in out:
        print(i.shape)
