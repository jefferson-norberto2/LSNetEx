import torch
import torch.nn as nn
import torch.nn.functional as F

# Define o Dynamic Convolution Module
class DynamicConv(nn.Module):
    def __init__(self, ksize):
        super(DynamicConv, self).__init__()
        self.ksize = ksize

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, device):
        B, C, H, W = x.shape

        # Extração de recursos
        x = nn.Conv2d(C, C * 4, self.ksize, padding=1).to(device)(x)
        x = nn.ReLU().to(device)(x)
        x = nn.Conv2d(C * 4, C * 16, self.ksize, padding=1).to(device)(x)
        x = nn.ReLU().to(device)(x)
        x = nn.Conv2d(C * 16, C, kernel_size=1).to(device)(x)

        # Geração de pesos dinâmicos
        pooled = self.global_avg_pool(x)  # (B, C, 1, 1)
        dynamic_weights = pooled.view(B, C, 1, 1)  # Manter dimensão dos canais

        # Convolução para ajustar os pesos dinâmicos
        dynamic_weights = nn.Conv2d(C, C * self.ksize * self.ksize, kernel_size=1).to(device)(dynamic_weights)
        dynamic_weights = dynamic_weights.view(B, C, self.ksize * self.ksize)


        # Extração de patches
        x_unfold = F.unfold(x, kernel_size=self.ksize, padding=self.ksize // 2)
        x_unfold = x_unfold.view(B, C, self.ksize * self.ksize, H, W)

        # Ajuste dos pesos dinâmicos
        dynamic_weights = dynamic_weights.view(B, C, self.ksize * self.ksize, 1, 1)
        x_unfold = x_unfold.unsqueeze(1)  # Expande para corresponder aos canais de saída

        # Multiplicação e soma ao longo dos patches
        x_dynamic = torch.sum(x_unfold * dynamic_weights, dim=3)
        x_dynamic = torch.sum(x_dynamic, dim=3)  # Reduz a dimensão dos canais de entrada

        x_dynamic = nn.Conv2d(x_dynamic.shape[1], C, kernel_size=1).to(device)(x_dynamic)

        # Normalização
        x_dynamic = nn.BatchNorm2d(C).to(device)(x_dynamic)  # Deve ser consistente com out_channels
        return x_dynamic


# Implementação do BBA com Dynamic Convolutions
def tensor_bound_with_cnn(img, ksize, dynamic_conv, device=0):
    """
    :param img: tensor, B*C*H*W
    :param ksize: int, tamanho do kernel
    :param dynamic_conv: instância do módulo DynamicConv
    :return: tensor, diferença entre inflação e corrosão
    """
    pad_value = int((ksize - 1) // 2)
    
    # Aplica padding na entrada
    img_pad = F.pad(img, pad=[pad_value, pad_value, pad_value, pad_value], mode='constant', value=0)
    
    # Calcula a inflação (máximo local) com convolução dinâmica
    inflation = dynamic_conv(img_pad, device)
    
    # Calcula a corrosão (mínimo local) com convolução dinâmica invertida
    corrosion = -dynamic_conv(-img_pad, device)
    
    # Retorna a diferença entre inflação e corrosão
    return inflation - corrosion


# Exemplo de uso
if __name__ == "__main__":
    B, C, H, W = 2, 3, 64, 64  # Dimensões da entrada
    ksize = 3  # Tamanho do kernel
    img = torch.randn(B, C, H, W)  # Entrada simulada

    # Instancia DynamicConv
    dynamic_conv = DynamicConv( ksize=ksize)

    # Aplica BBA com convoluções dinâmicas
    result = tensor_bound_with_cnn(img, ksize, dynamic_conv)
    print("Resultado:", result.shape)
