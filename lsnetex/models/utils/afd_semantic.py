from torch import sigmoid, sum, norm, div, pow
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d
from torch.nn.init import kaiming_normal_, constant_

class AFD_semantic(Module):
    """
    Attention-based Fusion and Distillation (AFD) module for semantic attention loss calculation.

    Attributes:
        attention (Sequential): Sequential module containing convolutional layers for attention calculation.
        avg_pool (AdaptiveAvgPool2d): Adaptive average pooling layer for spatial pooling.

    Methods:
        forward(fm_s: Tensor, fm_t: Tensor, eps: float = 1e-6) -> Tensor:
            Forward pass of the AFD_semantic module.

    """
    def __init__(self, in_channels, att_f):
        super(AFD_semantic, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = Sequential(*[
            Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True),
            ReLU(inplace=True),
            Conv2d(mid_channels, in_channels, 3, 1, 1, bias=True)
        ])
        self.avg_pool = AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):
        """
        Forward pass of the AFD_semantic module.

        Args:
            fm_s (Tensor): Feature map from the source domain.
            fm_t (Tensor): Feature map from the target domain.
            eps (float): Small value added to avoid division by zero.

        Returns:
            Tensor: Semantic attention loss.

        """
        fm_t_pooled = self.avg_pool(fm_t)
        rho = self.attention(fm_t_pooled)
        rho = sigmoid(rho)
        rho = rho.view(rho.size(0), -1)  # garante shape [B, C]
        rho = rho / sum(rho, dim=1, keepdim=True)

        fm_s_norm = norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = div(fm_s, fm_s_norm + eps)
        fm_t_norm = norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = div(fm_t, fm_t_norm + eps)

        loss = rho * pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss