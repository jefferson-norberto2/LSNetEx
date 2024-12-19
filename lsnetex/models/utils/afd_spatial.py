from torch import sigmoid, sum, norm, div, pow
from torch.nn import Module, Sequential, Conv2d
from torch.nn.init import kaiming_normal_, constant_

class AFD_spatial(Module):
    """
    Attention-based Fusion and Distillation (AFD) module for spatial attention loss calculation.

    Attributes:
        attention (Sequential): Sequential module containing convolutional layers for attention calculation.

    Methods:
        forward(fm_s: Tensor, fm_t: Tensor, eps: float = 1e-6) -> Tensor:
            Forward pass of the AFD_spatial module.

    """
    def __init__(self, in_channels):
        """
        Initializes the AFD_spatial module.

        Args:
            in_channels (int): Number of input channels.
        
        """
        super(AFD_spatial, self).__init__()

        self.attention = Sequential(*[
            Conv2d(in_channels, 1, 3, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):
        """
        Forward pass of the AFD_spatial module.

        Args:
            fm_s (Tensor): Feature map from the source domain.
            fm_t (Tensor): Feature map from the target domain.
            eps (float): Small value added to avoid division by zero.

        Returns:
            Tensor: Spatial attention loss.

        """
        rho = self.attention(fm_t)
        rho = sigmoid(rho)
        rho = rho / sum(rho, dim=(2,3), keepdim=True)

        fm_s_norm = norm(fm_s, dim=1, keepdim=True)
        fm_s = div(fm_s, fm_s_norm + eps)
        fm_t_norm = norm(fm_t, dim=1, keepdim=True)
        fm_t = div(fm_t, fm_t_norm + eps)
        loss = rho * pow(fm_s - fm_t, 2).mean(dim=1, keepdim=True)
        loss =sum(loss,dim=(2,3)).mean(0)
        return loss