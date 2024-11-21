from torch.nn import Module, MultiheadAttention

class CrossAttentionModule(Module):
    """
    Implements Cross-Attention using MultiheadAttention.
    """
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionModule, self).__init__()
        self.cross_attention = MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value):
        # Flatten spatial dimensions for attention
        b, c, h, w = query.shape
        query = query.flatten(2).transpose(1, 2)  # [B, H*W, C]
        key = key.flatten(2).transpose(1, 2)      # [B, H*W, C]
        value = value.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Apply cross-attention
        attn_output, _ = self.cross_attention(query, key, value)
        attn_output = attn_output.transpose(1, 2).view(b, c, h, w)  # Reshape back to [B, C, H, W]
        return attn_output