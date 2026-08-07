"""
model.py — TriChronos-50M
Encoder-only Transformer for probabilistic time-series forecasting.

Architecture
------------
  d_model  = 768
  n_layers = 6
  n_heads  = 12
  patch_size = 8

Each encoder block alternates between:
  1. Temporal self-attention   (within the sequence dimension)
  2. Group   attention         (across the batch dimension — cross-series)
  3. BitLinear FFN

All Q/K/V/O projections inside attention blocks use BitLinear.
Patch embedding and quantile output head stay in FP16.

Output: 21 quantile predictions (τ = 0.05, 0.10, …, 0.95 + 0.50 median)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlinear import BitLinear


# ---------------------------------------------------------------------------
# Quantile levels (21 values)
# ---------------------------------------------------------------------------

QUANTILE_LEVELS: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                                 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                 0.85, 0.9, 0.95, 0.025, 0.975]
N_QUANTILES: int = len(QUANTILE_LEVELS)   # 21


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Patch embedding (FP16, not quantized)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Linear projection of each patch into the model dimension.
    Stays in FP16 — this is the entry-point to the model and the
    quantization boundary.
    """

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model, bias=True)
        self.pos_enc = SinusoidalPE(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches : (B, n_patches, patch_size)
        x = self.proj(patches)          # (B, n_patches, d_model)
        x = self.pos_enc(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# BitLinear multi-head attention (shared by temporal + group variants)
# ---------------------------------------------------------------------------

class BitMHA(nn.Module):
    """
    Multi-head attention using BitLinear projections for Q, K, V, and O.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = BitLinear(d_model, d_model, bias=False)
        self.k_proj = BitLinear(d_model, d_model, bias=False)
        self.v_proj = BitLinear(d_model, d_model, bias=False)
        self.o_proj = BitLinear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # → (B, n_heads, T, d_head)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        q_in: torch.Tensor,          # query source  (B, Tq, d_model)
        kv_in: torch.Tensor,         # key/value src (B, Tk, d_model)
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.q_proj(q_in))
        K = self._split_heads(self.k_proj(kv_in))
        V = self._split_heads(self.v_proj(kv_in))

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)    # (B, n_heads, Tq, d_head)
        out = self._merge_heads(out)   # (B, Tq, d_model)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Group (cross-series) attention
# ---------------------------------------------------------------------------

class GroupAttention(nn.Module):
    """
    Cross-series attention: treats the batch dimension as the sequence
    dimension and performs attention across different series in the batch.

    For each patch position t, we gather the representation of that position
    across all B series in the batch and run attention over the B "tokens".

    Assumption: the batch is composed of related series (same dataset/subset).
    The data pipeline sorts samples by subset to honour this assumption.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = BitMHA(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, d_model)
        B, T, D = x.shape

        # Swap B and T: treat each patch position as an independent "sequence"
        # of B "tokens" (one per series in the batch).
        x_t = x.permute(1, 0, 2)            # (T, B, d_model)

        # Run attention over the B dimension for each of the T positions.
        # We process all T positions at once by temporarily reshaping:
        #   (T, B, D) → viewed as (T*1, B, D) so that the batch dim = T
        #   and the sequence dim = B.
        x_flat = x_t.reshape(T, B, D)       # (T, B, D) — B is seq len here

        out = self.mha(x_flat, x_flat)       # (T, B, D)
        out = self.norm(out + x_flat)        # residual + norm

        # Restore original layout
        return out.permute(1, 0, 2)          # (B, T, D)


# ---------------------------------------------------------------------------
# BitLinear FFN
# ---------------------------------------------------------------------------

class BitFFN(nn.Module):
    """Two-layer FFN with BitLinear and GELU activation."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = BitLinear(d_model, ffn_dim, bias=False)
        self.fc2 = BitLinear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Single encoder block with:
      1. Temporal self-attention  (within sequence)  + residual + norm
      2. Group attention          (across batch)      + residual + norm
      3. BitLinear FFN            + residual + norm
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        # Temporal attention
        self.temporal_attn = BitMHA(d_model, n_heads, dropout)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Group (cross-series) attention
        self.group_attn = GroupAttention(d_model, n_heads, dropout)

        # FFN
        self.ffn = BitFFN(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Temporal self-attention
        residual = x
        x = self.temporal_attn(x, x)
        x = self.temporal_norm(x + residual)

        # 2. Group attention (operates on batch axis — no residual shape conflict)
        x = self.group_attn(x)

        # 3. FFN
        x = self.ffn(x)

        return x


# ---------------------------------------------------------------------------
# Quantile output head (FP16)
# ---------------------------------------------------------------------------

class QuantileHead(nn.Module):
    """
    FP16 head that maps the pooled encoder representation to 21 quantile
    predictions over the forecast horizon.

    Input  : (B, d_model)     — mean-pooled encoder output
    Output : (B, horizon, 21) — quantile predictions per future timestep
    """

    def __init__(self, d_model: int, horizon: int, n_quantiles: int = N_QUANTILES):
        super().__init__()
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        # Residual projection — stays FP16
        self.proj = nn.Linear(d_model, horizon * n_quantiles, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, d_model)
        out = self.proj(x)                              # (B, horizon*n_quantiles)
        return out.view(x.size(0), self.horizon, self.n_quantiles)


# ---------------------------------------------------------------------------
# TriChronos — full model
# ---------------------------------------------------------------------------

class TriChronos(nn.Module):
    """
    TriChronos-0.1B: encoder-only Transformer for time-series forecasting.

    Config (hand-verified ~49.9M params)
    --------------------------------------
    d_model   = 768
    n_layers  = 6
    n_heads   = 12
    patch_size = 8
    ffn_dim   = 2304  (~3 × d_model; tuned to hit ~50M target)
    """

    def __init__(
        self,
        patch_size: int = 8,
        d_model: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        ffn_dim: int = 2304,
        horizon: int = 24,
        dropout: float = 0.1,
        n_quantiles: int = N_QUANTILES,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.horizon = horizon

        # Patch embedding (FP16)
        self.patch_embed = PatchEmbedding(patch_size, d_model)

        # Encoder stack
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Output head (FP16)
        self.quantile_head = QuantileHead(d_model, horizon, n_quantiles)

        # Initialise weights
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, BitLinear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patches : (B, n_patches, patch_size)  — Gold-stage output

        Returns
        -------
        quantiles : (B, horizon, n_quantiles)  — probabilistic forecast
        """
        # Patch embedding
        x = self.patch_embed(patches)    # (B, n_patches, d_model)

        # Encoder stack
        for block in self.encoder:
            x = block(x)
        x = self.encoder_norm(x)         # (B, n_patches, d_model)

        # Global mean pooling over patch dimension
        x = x.mean(dim=1)               # (B, d_model)

        # Quantile predictions
        return self.quantile_head(x)     # (B, horizon, n_quantiles)

    # ------------------------------------------------------------------

    def count_params(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        params = (
            self.parameters() if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------

    @property
    def quantile_levels(self) -> List[float]:
        return QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Quick smoke test (python model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    model = TriChronos()
    n_params = model.count_params()
    print(f"TriChronos-50M - {n_params:,} trainable parameters")
    print(f"  Target: ~49,900,000")
    print(f"  Delta : {abs(n_params - 49_900_000):,}")

    # Synthetic batch
    B, T = 4, 64    # batch=4, 64 patches
    patches = torch.randn(B, T, model.patch_size)
    quantiles = model(patches)

    print(f"\nInput   shape : {patches.shape}")
    print(f"Output  shape : {quantiles.shape}   (expect ({B}, {model.horizon}, {N_QUANTILES}))")
    assert quantiles.shape == (B, model.horizon, N_QUANTILES)

    # Backward pass
    loss = quantiles.sum()
    loss.backward()
    print(f"\nBackward pass — OK")
    print("model.py - all checks passed OK")
