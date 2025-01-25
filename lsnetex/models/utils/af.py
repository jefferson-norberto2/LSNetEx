import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionFusion, self).__init__()
        # Camadas para calcular os pesos de atenção
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F6, F7):
        # Passo 1: Concatenar as features
        combined = torch.cat((F6, F7), dim=1)

        # Passo 2: Calcular os pesos de atenção
        attention = self.global_avg_pool(combined)  # Pooling global
        attention = F.relu(self.conv1(attention))     # Redução de dimensionalidade
        attention = self.sigmoid(self.conv2(attention))  # Calcular pesos normalizados (0 a 1)

        # Passo 3: Criar atenções separadas para F6 e F7
        attention_F6 = attention[:, :F6.size(1), :, :]  # Atenção para F6
        attention_F6 = attention_F6.expand_as(F6)  # Expandir para as dimensões de F6

        attention_F7 = attention[:, F6.size(1):, :, :]  # Atenção para F7
        attention_F7 = attention_F7.expand_as(F7)  # Expandir para as dimensões de F7

        # Passo 3: Aplicar os pesos às features
        F6_weighted = attention_F6 * F6
        F7_weighted = (1 - attention_F7) * F7

        # Passo 4: Combinar as features ponderadas
        fused = torch.cat((F6_weighted, F7_weighted), dim=1)
        return fused

if __name__ == '__main__':
    # Exemplo de entrada
    F6 = torch.rand(1, 320, 32, 32)  # Batch size 1, 320 canais, 32x32 spatial size
    F7 = torch.rand(1, 320, 32, 32)

    # Inicializando a fusão com atenção espacial
    fusion_layer = AttentionFusion(in_channels=640)

    # Realizando a fusão
    fused_output = fusion_layer(F6, F7)
    print(f"Saída após fusão: {fused_output.shape}")