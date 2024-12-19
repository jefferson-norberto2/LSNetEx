# set loss function
from torch.nn import BCEWithLogitsLoss, Module
from torch import sigmoid

class IOUBCE_loss(Module):
    def __init__(self):
        super(IOUBCE_loss, self).__init__()
        self.nll_lose = BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            bce = self.nll_lose(inputs,targets)
            pred = sigmoid(inputs)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b