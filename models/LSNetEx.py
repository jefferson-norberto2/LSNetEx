from torch import cat
from torch.nn import Module, Sequential, Conv2d, UpsamplingBilinear2d, GELU, BatchNorm2d
from models.afd_semantic import AFD_semantic
from models.afd_spatial import AFD_spatial
from models.mobilenetv3 import mobilenet_v3_large

class LSNetEx(Module):
    """
    LSNet (Light Semantic Network) model for RGB and depth/thermal image fusion.

    Attributes:
        rgb_pretrained (MobileNetV3Ex): Pre-trained MobileNetV3 large model for RGB input.
        depth_pretrained (MobileNetV3Ex): Pre-trained MobileNetV3 large model for depth input.
        upsample1_g to upsample7_g (Sequential): Upsampling layers for feature fusion.
        conv_g, conv2_g, conv3_g (Conv2d): Convolutional layers for final and intermediate output.
        AFD_semantic_7_R_T to AFD_spatial_1_R_T: Optional modules for semantic and spatial attention loss calculation (used during training).

    Methods:
        forward(rgb: Tensor, ti: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]: Forward pass of the LSNet model.

    """
    def __init__(self):
        super(LSNetEx, self).__init__()
        print('LsNet - V3LargeTL')
        self.rgb_pretrained = mobilenet_v3_large(pretrained=True)
        self.depth_pretrained = mobilenet_v3_large(pretrained=True)

        # Upsample_model
        self.upsample1_g = Sequential(Conv2d(76, 38, 3, 1, 1, ), BatchNorm2d(38), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_g = Sequential(Conv2d(118, 60, 3, 1, 1, ), BatchNorm2d(60), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_g = Sequential(Conv2d(188, 94, 3, 1, 1, ), BatchNorm2d(94), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = Sequential(Conv2d(296, 148, 3, 1, 1, ), BatchNorm2d(148), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample5_g = Sequential(Conv2d(432, 216, 3, 1, 1, ), BatchNorm2d(216), GELU(),
                                         UpsamplingBilinear2d(scale_factor=1, ))
        
        self.upsample6_g = Sequential(Conv2d(640, 320, 3, 1, 1, ), BatchNorm2d(320), GELU(),  UpsamplingBilinear2d(scale_factor=2, ))
        
        self.upsample7_g = Sequential(Conv2d(960, 480, 3, 1, 1, ), BatchNorm2d(480), GELU())
        
        self.conv_g = Conv2d(38, 1, 1)
        self.conv2_g = Conv2d(60, 1, 1)
        self.conv3_g = Conv2d(94, 1, 1)

        # Tips: speed test and params and more this part is not included.
        # please comment this part when involved.
        if self.training:
            self.AFD_semantic_7_R_T = AFD_semantic(960, 0.0625)
            self.AFD_semantic_6_R_T = AFD_semantic(160, 0.0625)
            self.AFD_semantic_5_R_T = AFD_semantic(112, 0.0625)
            self.AFD_semantic_4_R_T = AFD_semantic(80, 0.0625)
            self.AFD_spatial_4_R_T = AFD_spatial(80)
            self.AFD_spatial_3_R_T = AFD_spatial(40)
            self.AFD_spatial_2_R_T = AFD_spatial(24)
            self.AFD_spatial_1_R_T = AFD_spatial(16)

    def forward(self, rgb, ti):
        """
        Forward pass of the LSNet model.

        Args:
            rgb (Tensor): Input RGB tensor.
            ti (Tensor): Input tensor from a different modality (e.g., depth).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Output tensors from the model and additional losses (if in training mode).

        """

        A1, A2, A3, A4, A5, A6, A7 = self.rgb_pretrained(rgb)
        A1_t, A2_t, A3_t, A4_t, A5_t, A6_t, A7_t = self.depth_pretrained(ti)

        F7 = A7_t + A7
        F6 = A6_t + A6
        F5 = A5_t + A5
        F4 = A4_t + A4
        F3 = A3_t + A3
        F2 = A2_t + A2
        F1 = A1_t + A1

        F7 = self.upsample7_g(F7)
        F6 = cat((F6, F7), dim=1)
        F6 = self.upsample6_g(F6)
        F5 = cat((F5, F6), dim=1)
        F5 = self.upsample5_g(F5)
        F4 = cat((F4, F5), dim=1)
        F4 = self.upsample4_g(F4)
        F3 = cat((F3, F4), dim=1)
        F3 = self.upsample3_g(F3)
        F2 = cat((F2, F3), dim=1)
        F2 = self.upsample2_g(F2)
        F1 = cat((F1, F2), dim=1)
        F1 = self.upsample1_g(F1)

        out = self.conv_g(F1)

        if self.training:
            out3 = self.conv3_g(F3)
            out2 = self.conv2_g(F2)
            loss_semantic_7_R_T = self.AFD_semantic_7_R_T(A7, A7_t.detach())
            loss_semantic_7_T_R = self.AFD_semantic_7_R_T(A7_t, A7.detach())
            loss_semantic_6_R_T = self.AFD_semantic_6_R_T(A6, A6_t.detach())
            loss_semantic_6_T_R = self.AFD_semantic_6_R_T(A6_t, A6.detach())
            loss_semantic_5_R_T = self.AFD_semantic_5_R_T(A5, A5_t.detach())
            loss_semantic_5_T_R = self.AFD_semantic_5_R_T(A5_t, A5.detach())
            loss_semantic_4_R_T = self.AFD_semantic_4_R_T(A4, A4_t.detach())
            loss_semantic_4_T_R = self.AFD_semantic_4_R_T(A4_t, A4.detach())
            loss_spatial_4_R_T = self.AFD_spatial_4_R_T(A4, A4_t.detach())
            loss_spatial_4_T_R = self.AFD_spatial_4_R_T(A4_t, A4.detach())
            loss_spatial_3_R_T = self.AFD_spatial_3_R_T(A3, A3_t.detach())
            loss_spatial_3_T_R = self.AFD_spatial_3_R_T(A3_t, A3.detach())
            loss_spatial_2_R_T = self.AFD_spatial_2_R_T(A2, A2_t.detach())
            loss_spatial_2_T_R = self.AFD_spatial_2_R_T(A2_t, A2.detach())
            loss_spatial_1_R_T = self.AFD_spatial_1_R_T(A1, A1_t.detach())
            loss_spatial_1_T_R = self.AFD_spatial_1_R_T(A1_t, A1.detach())
            loss_KD = loss_semantic_7_R_T + loss_semantic_7_T_R + \
                      loss_semantic_6_R_T + loss_semantic_6_T_R + \
                      loss_semantic_5_R_T + loss_semantic_5_T_R + \
                      loss_semantic_4_R_T + loss_semantic_4_T_R + \
                      loss_spatial_4_R_T + loss_spatial_4_T_R + \
                      loss_spatial_3_R_T + loss_spatial_3_T_R + \
                      loss_spatial_2_R_T + loss_spatial_2_T_R + \
                      loss_spatial_1_R_T + loss_spatial_1_T_R
            return out, out2, out3, loss_KD
        
        return out        
