from torch import sigmoid, sum, norm, div, pow, cat
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, UpsamplingBilinear2d, GELU, BatchNorm2d
from torch.nn.init import kaiming_normal_, constant_
from models.mobilenetv3 import mobilenet_v3_small_ex, mobilenet_v3_large_ex

class AFD_semantic(Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

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

        fm_t_pooled = self.avg_pool(fm_t)
        rho = self.attention(fm_t_pooled)
        rho = sigmoid(rho.squeeze())
        rho = rho / sum(rho, dim=1, keepdim=True)

        fm_s_norm = norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = div(fm_s, fm_s_norm + eps)
        fm_t_norm = norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = div(fm_t, fm_t_norm + eps)

        loss = rho * pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss


class AFD_spatial(Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels):
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

class LSNet(Module):
    def __init__(self):
        super(LSNet, self).__init__()
        # rgb,depth encode
        self.rgb_pretrained = mobilenet_v3_large_ex(pretrained=True)
        self.depth_pretrained = mobilenet_v3_large_ex(pretrained=True)

        # Upsample_model
        self.upsample1_g = Sequential(Conv2d(68, 34, 3, 1, 1, ), BatchNorm2d(34), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_g = Sequential(Conv2d(104, 52, 3, 1, 1, ), BatchNorm2d(52), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_g = Sequential(Conv2d(168, 80, 3, 1, 1, ), BatchNorm2d(80), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = Sequential(Conv2d(208, 128, 3, 1, 1, ), BatchNorm2d(128), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))
        
        self.upsample5_g = Sequential(Conv2d(160, 128, 3, 1, 1, ), BatchNorm2d(128), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))
        
        self.upsample6_g = Sequential(Conv2d(960, 168, 3, 1, 1, ), BatchNorm2d(168), GELU(),
                                         UpsamplingBilinear2d(scale_factor=2, ))



        self.conv_g = Conv2d(34, 1, 1)
        self.conv2_g = Conv2d(52, 1, 1)
        self.conv3_g = Conv2d(80, 1, 1)


        # Tips: speed test and params and more this part is not included.
        # please comment this part when involved.
        if self.training:
            self.AFD_semantic_6_R_T = AFD_semantic(960, 0.0625)
            self.AFD_semantic_5_R_T = AFD_semantic(160, 0.0625)
            self.AFD_semantic_4_R_T = AFD_semantic(80, 0.0625)
            self.AFD_semantic_3_R_T = AFD_semantic(40, 0.0625)
            self.AFD_spatial_3_R_T = AFD_spatial(40)
            self.AFD_spatial_2_R_T = AFD_spatial(24)
            self.AFD_spatial_1_R_T = AFD_spatial(16)


    def forward(self, rgb, ti):
        # rgb
        A1, A2, A3, A4, A5, A6 = self.rgb_pretrained(rgb)
        # ti
        A1_t, A2_t, A3_t, A4_t, A5_t, A6_t = self.depth_pretrained(ti)

        F6 = A6_t + A6
        F5 = A5_t + A5
        F4 = A4_t + A4
        F3 = A3_t + A3
        F2 = A2_t + A2
        F1 = A1_t + A1

        
        F6 = self.upsample6_g(F6)
        #F5 = cat((F5, F6), dim=1)
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
            loss_semantic_6_R_T = self.AFD_semantic_6_R_T(A6, A6_t.detach())
            loss_semantic_6_T_R = self.AFD_semantic_6_R_T(A6_t, A6.detach())
            loss_semantic_5_R_T = self.AFD_semantic_5_R_T(A5, A5_t.detach())
            loss_semantic_5_T_R = self.AFD_semantic_5_R_T(A5_t, A5.detach())
            loss_semantic_4_R_T = self.AFD_semantic_4_R_T(A4, A4_t.detach())
            loss_semantic_4_T_R = self.AFD_semantic_4_R_T(A4_t, A4.detach())
            loss_semantic_3_R_T = self.AFD_semantic_3_R_T(A3, A3_t.detach())
            loss_semantic_3_T_R = self.AFD_semantic_3_R_T(A3_t, A3.detach())
            loss_spatial_3_R_T = self.AFD_spatial_3_R_T(A3, A3_t.detach())
            loss_spatial_3_T_R = self.AFD_spatial_3_R_T(A3_t, A3.detach())
            loss_spatial_2_R_T = self.AFD_spatial_2_R_T(A2, A2_t.detach())
            loss_spatial_2_T_R = self.AFD_spatial_2_R_T(A2_t, A2.detach())
            loss_spatial_1_R_T = self.AFD_spatial_1_R_T(A1, A1_t.detach())
            loss_spatial_1_T_R = self.AFD_spatial_1_R_T(A1_t, A1.detach())
            loss_KD = loss_semantic_6_R_T + loss_semantic_6_T_R + \
                      loss_semantic_5_R_T + loss_semantic_5_T_R + \
                      loss_semantic_4_R_T + loss_semantic_4_T_R + \
                      loss_semantic_3_R_T + loss_semantic_3_T_R + \
                      loss_spatial_3_R_T + loss_spatial_3_T_R + \
                      loss_spatial_2_R_T + loss_spatial_2_T_R + \
                      loss_spatial_1_R_T + loss_spatial_1_T_R
            return out, out2, out3, loss_KD
        return out

