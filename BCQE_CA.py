import torch
import torch.nn as nn
import torch.nn.functional as F

class CirculantMatrix(nn.Module):
    # 循环矩阵实现

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.dim() == 2:  # (B, dim)
            B, d = x.shape
            idx = torch.arange(d, device=x.device)
            shift_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)) % d
            return x[:, shift_matrix]  # (B, d, d)
        else:  # (B, N, dim)
            B, N, d = x.shape
            idx = torch.arange(d, device=x.device)
            shift_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)) % d
            return x[:, :, shift_matrix]  # (B, N, d, d)


class CirculantCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 基础投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 仅为 Q 准备循环矩阵模块
        self.circulant = CirculantMatrix(dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def apply_q_circulant(self, q):

        B, N, C = q.shape
        q_global = q.mean(dim=1)  # (B, C)
        circ_matrix = self.circulant(q_global)  # (B, C, C)

        # 应用循环变换
        transformed_q = torch.bmm(q.reshape(B * N, 1, C),
                                  circ_matrix.repeat_interleave(N, dim=0)).reshape(B, N, C)
        
        return q + transformed_q


    def forward(self, query, key, value, mask=None, return_attention=False):
        B, N_q, C = query.shape
        B, N_k, C = key.shape

        # 投影 Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 仅对 Q 做循环变换
        Q = self.apply_q_circulant(Q)

        # Multi-head 切分
        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 标准 Attention 计算
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出投影
        out = (attn @ V).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out


class MultiModalCCABlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # 验证维度兼容性
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        
        self.dim = dim
        self.norm1_v = nn.LayerNorm(dim)
        self.norm1_t = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim * 2)

        # 双向 Cross-Attention
        self.v2t_attn = CirculantCrossAttention(
            dim, num_heads, dropout=dropout
        )
        self.t2v_attn = CirculantCrossAttention(
            dim, num_heads, dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim * 2),
            nn.Dropout(dropout)
        )

    def forward(self, vision_feat, text_feat):
        v_norm = self.norm1_v(vision_feat)
        t_norm = self.norm1_t(text_feat)

        v_attended = self.v2t_attn(v_norm, t_norm, t_norm)
        t_attended = self.t2v_attn(t_norm, v_norm, v_norm)

        vision_feat = vision_feat + v_attended
        text_feat = text_feat + t_attended

        fused = torch.cat([vision_feat, text_feat], dim=-1)  # [B, N, dim*2]
        fused = fused + self.mlp(self.norm2(fused))

        return fused