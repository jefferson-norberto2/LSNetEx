from torch.nn import Module, BCELoss

class IOUBCEWithoutLogits_loss(Module):
    def __init__(self):
        super(IOUBCEWithoutLogits_loss, self).__init__()
        self.nll_lose = BCELoss()

    def forward(self, input_scale, target_scale):
        b,c,h,w = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, target_scale):

            bce = self.nll_lose(inputs,targets)

            inter = (inputs * targets).sum(dim=(1, 2))
            union = (inputs + targets).sum(dim=(1, 2))
            IOU = (inter + 1) / (union - inter + 1)
            loss.append(1- IOU + bce)
        total_loss = sum(loss)
        return total_loss / b