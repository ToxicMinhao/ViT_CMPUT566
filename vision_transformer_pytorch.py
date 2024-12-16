import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, dim):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

    def forward(self, x):
        return x + self.positional_embedding

class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout_rate=0.1):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, mlp_dim, num_heads, dropout_rate=0.1, attention_dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attention_dropout_rate)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, mlp_dim, dropout_rate)

    def forward(self, x):
        # Attention block with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_output)
        # MLP block with residual connection
        x = self.ln2(x + self.mlp(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, mlp_dim, num_heads, channels=3, dropout_rate=0.1, attention_dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(self.patch_dim, dim),
        )
        
        self.pos_embedding = PositionalEmbedding(self.num_patches + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.encoder = nn.Sequential(*[
            TransformerEncoder(dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            for _ in range(depth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])
