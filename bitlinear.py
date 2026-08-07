"""
bitlinear.py — TriChronos-0.1B
Absmean ternary weight quantization (BitNet-style) with Straight-Through Estimator.
Per-token activation quantization applied before the linear projection.

Reference: "The Era of 1-bit LLMs" (Ma et al., 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------

def absmean_quantize(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Absmean ternary quantization.

    1. Compute the scale γ = mean(|W|)  (a single scalar per weight matrix)
    2. Divide W by γ, round to nearest integer, clip to {-1, 0, +1}

    During the forward pass only the ternary version is used; gradients flow
    through unmodified (Straight-Through Estimator applied at call site).
    """
    gamma = W.abs().mean().clamp(min=eps)
    W_scaled = W / gamma
    W_ternary = W_scaled.round().clamp(-1, 1)
    return W_ternary, gamma


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------

def per_token_quant(x: torch.Tensor, bits: int = 8, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric per-token (per-row) quantization of activations to `bits` bits.

    x : (..., d_model)  — last dim is the feature dimension, all other dims
                          are treated as independent "tokens".

    Each token is scaled independently so that its dynamic range fills the
    integer grid [-2^(bits-1)+1, 2^(bits-1)-1].  We use absmax scaling
    (same approach as LLM.int8()).
    """
    qmax = 2 ** (bits - 1) - 1                       # 127 for 8-bit
    # absmax per token; keep dims for broadcasting
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    x_scaled = (x / scale * qmax).round().clamp(-qmax, qmax)
    # dequantize back to float so the rest of the graph stays in FP16/BF16
    x_dequant = x_scaled / qmax * scale
    return x_dequant


# ---------------------------------------------------------------------------
# BitLinear layer
# ---------------------------------------------------------------------------

class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear that:
      • Keeps FP16/BF16 master weights (for optimizer stability)
      • Quantizes weights to ternary {-1, 0, +1} during the forward pass
      • Applies per-token activation quantization to the input
      • Uses STE: gradients pass through the quantization step unchanged

    Usage is identical to nn.Linear:
        layer = BitLinear(in_features, out_features, bias=False)
        y = layer(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,          # bias=False is the BitNet default
        act_bits: int = 8,
        weight_eps: float = 1e-6,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.act_bits = act_bits
        self.weight_eps = weight_eps

        # LayerNorm before quantization stabilises activations (BitNet paper §3)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Normalise activations
        x = self.norm(x)

        # 2. Quantize activations per-token (straight through — no rounding
        #    in the backward graph, we use the dequantized value directly)
        x_q = per_token_quant(x, bits=self.act_bits, eps=self.weight_eps)

        # 3. Quantize weights with STE
        #    Forward: use ternary weights scaled by γ
        #    Backward: gradient flows through as if weights were continuous
        W_ternary, gamma = absmean_quantize(self.weight, eps=self.weight_eps)
        # STE: detach the rounding error so gradients see a straight-through copy
        W_ste = self.weight + (W_ternary - self.weight).detach()
        W_ste_scaled = W_ste * gamma                  # restore scale for correct output magnitude

        return F.linear(x_q, W_ste_scaled, self.bias)

    # ------------------------------------------------------------------
    # Convenience: export ternary weights for storage (packed int8)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ternary_weights(self) -> torch.Tensor:
        """Return quantized weights as int8 tensor (values in {-1, 0, 1})."""
        W_ternary, _ = absmean_quantize(self.weight, eps=self.weight_eps)
        return W_ternary.to(torch.int8)

    @torch.no_grad()
    def weight_scale(self) -> torch.Tensor:
        """Return the absmean scale γ used during quantization."""
        _, gamma = absmean_quantize(self.weight, eps=self.weight_eps)
        return gamma


# ---------------------------------------------------------------------------
# Quick sanity check (run directly: python bitlinear.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    layer = BitLinear(64, 128)
    x = torch.randn(4, 16, 64)          # (batch, seq_len, d_model)
    y = layer(x)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {y.shape}")
    assert y.shape == (4, 16, 128), "Unexpected output shape"

    # Verify STE: backward pass should not raise
    loss = y.sum()
    loss.backward()
    print(f"Weight grad  : {layer.weight.grad.shape}  (STE working)")

    # Ternary check
    W_t = layer.ternary_weights()
    unique_vals = W_t.unique().tolist()
    print(f"Ternary unique values: {unique_vals}  (expected subset of [-1, 0, 1])")
    assert all(v in (-1, 0, 1) for v in unique_vals)

    print("bitlinear.py - all checks passed OK")
