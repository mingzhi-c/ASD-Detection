import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # B L D
        x = x + self.pe[0: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class AudioEncoder(nn.Module):
    def __init__(self, dim=256, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = PositionalEncoding(d_model=dim)

    def forward(self, x):
        window = torch.hann_window(1024, device=x.device)
        x = x.squeeze(1)
        stft = torch.stft(x, n_fft=1024, hop_length=256, win_length=1024, window=window, center=True,
                          pad_mode='reflect', return_complex=True)
        x = torch.abs(stft)
        x = torch.log1p(x)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
        x = x.unsqueeze(1)
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        # Positional Encoding
        x = self.positional_encoding(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.up_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.down_proj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(dim=dim, n_heads=num_heads)
        self.mlp = MLP(dim=dim)
        # Normalization before self attention
        self.input_norm = RMSNorm(dim)
        # Normalization before mlp
        self.post_attention_layernorm = RMSNorm(dim)

    def forward(self, x):
        h = x + self.self_attn(self.input_norm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(self, x):
        batch_size, _, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        xk = xk.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        xv = xv.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        output = F.scaled_dot_product_attention(xq, xk, xv).permute(0, 2, 1, 3)
        output = output.flatten(-2)
        return self.wo(output)

class AudioClassifier(nn.Module):
    def __init__(self, patch_size=32, dim=256, num_heads=16, n_layers=4, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.encoder = AudioEncoder(patch_size=patch_size, dim=dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads) for _ in range(n_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.ln_post = nn.LayerNorm(dim)
        self.classification_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        pred = self.classification_head(x[:, 0])
        return pred

if __name__ == "__main__":
    x = torch.randn(1, 1, 16000)
    model = AudioClassifier()
    y = model(x)
    print(y.shape)