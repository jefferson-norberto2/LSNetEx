from torch import min, max, median
from torch.nn.functional import pad

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

# BBA
def tesnor_bound(img, ksize):

    '''
    :param img: tensor, B*C*H*W
    :param ksize: tensor, ksize * ksize
    :param 2patches: tensor, B * C * H * W * ksize * ksize
    :return: tensor, (inflation - corrosion), B * C * H * W
    '''

    B, C, H, W = img.shape
    pad_value = int((ksize - 1) // 2)
    img_pad = pad(img, pad=[pad_value, pad_value, pad_value, pad_value], mode='constant',value = 0)
    # unfold in the second and third dimensions
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    corrosion, _ = min(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    inflation, _ = max(patches.contiguous().view(B, C, H, W, -1), dim=-1)
    enffort, _ = median(patches.contiguous().view(B, C, H, W, -1), dim=-1)

    inflation = inflation.contiguous()
    corrosion = corrosion.contiguous()

    result = inflation - corrosion + enffort
    result = (result - result.min()) / (result.max() - result.min())    
    return result

import torch.nn as nn
import torch.nn.functional as F

class ConvPatchEnhancer(nn.Module):
    def __init__(self, ksize):
        super(ConvPatchEnhancer, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 input channel, 8 filters
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=ksize, padding=0)  # Output matches patch size
        self.activation = nn.Sigmoid()  # Ensure outputs are in range [0, 1]

    def forward(self, x):
        # x: [B, C, H, W, patch_size, patch_size]
        B, C, H, W, patch_h, patch_w = x.shape
        x = x.reshape(B * C * H * W, 1, patch_h, patch_w)  # Treat patches as separate inputs for CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # Output shape: [B*C*H*W, 1, 1, 1]
        x = self.activation(x)
        return x.view(B, C, H, W)  # Reshape back to original structure


def tensor_bound_with_cnn(img, ksize, model, device=0):
    '''
    :param img: tensor, B*C*H*W
    :param ksize: int, size of the patch
    :param model: nn.Module, CNN to process patches
    :param device: int, device for computation
    :return: tensor, output processed by the CNN, B*C*H*W
    '''
    B, C, H, W = img.shape
    pad_value = int((ksize - 1) // 2)
    img_pad = F.pad(img, pad=[pad_value, pad_value, pad_value, pad_value], mode='constant', value=0)

    # Unfold into patches
    patches = img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)  # [B, C, H, W, ksize, ksize]

    # Pass patches through the CNN
    output = model(patches)  # [B, C, H, W]
    return output

