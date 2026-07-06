"""
Cross-modal multi-head attention module.

- Accepts vector or sequence inputs:
  q: (B, Dq) or (B, Lq, Dq)
  k: (B, Dkv) or (B, Lk, Dkv)
  v: (B, Dkv) or (B, Lv, Dkv)
- Projects to common dim_model, splits into n_heads, applies scaled dot-product attention.
- Returns:
  - (B, dim_model) if Lq == 1 (vector input)
  - (B, Lq, dim_model) otherwise
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_model: int = 512,
        n_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim_model % n_heads == 0, "dim_model must be divisible by n_heads"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.d_head = dim_model // n_heads

        # Linear projections
        self.q_proj = nn.Linear(dim_q, dim_model, bias=True)
        self.k_proj = nn.Linear(dim_kv, dim_model, bias=True)
        self.v_proj = nn.Linear(dim_kv, dim_model, bias=True)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=True)
        self.out_drop = nn.Dropout(proj_dropout)

    def _as_seq(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Ensure x has shape (B, L, D). If original was (B, D), returns (B, 1, D) and flag True.
        """
        if x.dim() == 2:
            return x.unsqueeze(1), True
        elif x.dim() == 3:
            return x, False
        else:
            raise ValueError(f"Expected x with dim 2 or 3, got shape={tuple(x.shape)}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: (B, Dq) or (B, Lq, Dq)
            k: (B, Dkv) or (B, Lk, Dkv)
            v: (B, Dkv) or (B, Lv, Dkv)
            key_padding_mask: optional bool mask of shape (B, Lk), True where to mask (exclude)
        Returns:
            fused:
              - (B, dim_model) if input q was (B, Dq)
              - (B, Lq, dim_model) if input q was (B, Lq, Dq)
        """
        B = q.shape[0]
        q_seq, q_was_vec = self._as_seq(q)  # (B, Lq, Dq)
        k_seq, _ = self._as_seq(k)          # (B, Lk, Dkv)
        v_seq, _ = self._as_seq(v)          # (B, Lv, Dkv)
        assert k_seq.size(1) == v_seq.size(1), "Key and Value must have same sequence length"

        # Linear projections
        Q = self.q_proj(q_seq)  # (B, Lq, dim_model)
        K = self.k_proj(k_seq)  # (B, Lk, dim_model)
        V = self.v_proj(v_seq)  # (B, Lk, dim_model)

        # Split heads
        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # (B, L, dim_model) -> (B, n_heads, L, d_head)
            return x.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Scaled dot-product attention
        scores = torch.matmul(Qh, Kh.transpose(-2, -1))  # (B, h, Lq, Lk)
        scores = scores / math.sqrt(self.d_head)

        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True means mask out
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,Lk)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B, h, Lq, Lk)
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, Vh)  # (B, h, Lq, d_head)
        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, -1, self.dim_model)  # (B, Lq, dim_model)

        out = self.out_proj(context)  # (B, Lq, dim_model)
        out = self.out_drop(out)

        if q_was_vec:
            return out.squeeze(1)  # (B, dim_model)
        return out  # (B, Lq, dim_model)


class HierarchicalTriModalFusion(nn.Module):
    """
    Hierarchical Tri-Modal Fusion:
    1) Structure-level fusion: PhysChem <-> SecondaryStructure (two-way cross-modal attention + residual MLP)
    2) Final fusion: ESM-2 <-> (Phys+SS fused) (two-way cross-modal attention + residual MLP)
    Returns: (B, dim_model)
    """
    def __init__(
        self,
        dim_esm: int,
        dim_phys: int,
        dim_ss: int,
        dim_model: int = 512,
        n_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # level-1: Phys <-> SS
        self.cma_p2s = CrossModalAttention(dim_q=dim_phys, dim_kv=dim_ss, dim_model=dim_model, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.cma_s2p = CrossModalAttention(dim_q=dim_ss, dim_kv=dim_phys, dim_model=dim_model, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.ln_struct = nn.LayerNorm(dim_model)
        self.mlp_struct = nn.Sequential(
            nn.Linear(dim_model, int(dim_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim_model * mlp_ratio), dim_model),
            nn.Dropout(dropout),
        )
        # level-2: ESM <-> fused(Phys+SS)
        self.cma_e2struct = CrossModalAttention(dim_q=dim_esm, dim_kv=dim_model, dim_model=dim_model, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.cma_struct2e = CrossModalAttention(dim_q=dim_model, dim_kv=dim_esm, dim_model=dim_model, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.ln_final = nn.LayerNorm(dim_model)
        self.mlp_final = nn.Sequential(
            nn.Linear(dim_model, int(dim_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim_model * mlp_ratio), dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, esm_vecs: torch.Tensor, phys_vecs: torch.Tensor, ss_vecs: torch.Tensor) -> torch.Tensor:
        # Level-1: PhysChem <-> SecondaryStructure
        ps = 0.5 * (self.cma_p2s(phys_vecs, ss_vecs, ss_vecs) + self.cma_s2p(ss_vecs, phys_vecs, phys_vecs))  # (B, dim_model)
        ps = ps + self.mlp_struct(self.ln_struct(ps))  # residual

        # Level-2: ESM <-> (Phys+SS)
        fused = 0.5 * (self.cma_e2struct(esm_vecs, ps, ps) + self.cma_struct2e(ps, esm_vecs, esm_vecs))  # (B, dim_model)
        fused = fused + self.mlp_final(self.ln_final(fused))  # residual
        return fused