import torch.nn as nn
import torch.nn.functional as F

class DynamicDilateErode(nn.Module):
    def __init__(self, in_channels, kernel_size, num_dynamic_kernels=4):
        super(DynamicDilateErode, self).__init__()
        self.kernel_size = kernel_size
        self.num_dynamic_kernels = num_dynamic_kernels

        # Para calcular os filtros dinâmicos de dilatação
        self.dynamic_dilate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_dynamic_kernels * kernel_size * kernel_size, kernel_size=1),
            nn.ReLU()
        )
        
        # Para calcular os filtros dinâmicos de erosão
        self.dynamic_erode = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_dynamic_kernels * kernel_size * kernel_size, kernel_size=1),
            nn.ReLU()
        )

        # Seleção dos pesos de dilatação e erosão
        self.kernel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_dynamic_kernels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Gerar pesos para dilatação
        dilate_kernels = self.dynamic_dilate(x)
        dilate_kernels = dilate_kernels.view(batch_size, self.num_dynamic_kernels, 1, self.kernel_size, self.kernel_size)

        # Gerar pesos para erosão
        erode_kernels = self.dynamic_erode(x)
        erode_kernels = erode_kernels.view(batch_size, self.num_dynamic_kernels, 1, self.kernel_size, self.kernel_size)

        # Selecionar pesos de acordo com o contexto
        kernel_weights = self.kernel_selector(x).view(batch_size, self.num_dynamic_kernels, 1, 1, 1)
        dilate_kernels = (dilate_kernels * kernel_weights).sum(dim=1)
        erode_kernels = (erode_kernels * kernel_weights).sum(dim=1)

        # Aplicar convoluções dinâmicas para dilatação e erosão
        dilated = F.conv2d(x, dilate_kernels, stride=1, padding=self.kernel_size // 2, groups=1)
        eroded = F.conv2d(x, erode_kernels, stride=1, padding=self.kernel_size // 2, groups=1)

        return dilated, eroded
