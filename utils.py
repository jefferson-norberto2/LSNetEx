from torch import min, max
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
    return inflation - corrosion