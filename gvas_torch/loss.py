"""
loss.py
=======
Loss functions and regularisers.

Pearson correlation for complex fields uses the real part of the
centred Hermitian inner product as the numerator (which is the
canonical generalisation); the legacy code used ``abs(sum(...))``
which drops the sign of the correlation. If you need to reproduce
legacy numbers exactly set ``pearson_variant='legacy'``.
"""
from __future__ import annotations

from typing import Literal

import torch

from .config import Config


# ---------------------------------------------------------------------------
# Fidelity losses
# ---------------------------------------------------------------------------

def mse_complex(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude of the residual: ``mean |pred - target|^2``.

    Works for real or complex inputs; identical to L2 when inputs are real.
    """
    return torch.mean((pred - target).abs() ** 2)


def _hermitian_variance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Centred Hermitian inner product per batch element.

    Returns a tensor of shape ``(B, 1, 1)``. Reducing over the last two
    spatial dims only, so an outer ``mean`` still makes sense for
    batched loss computation.
    """
    a_mean = a.mean(dim=(-2, -1), keepdim=True)
    b_mean = b.mean(dim=(-2, -1), keepdim=True)
    return torch.sum((a - a_mean) * (b - b_mean).conj(), dim=(-2, -1), keepdim=True)


def pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    variant: Literal["real", "abs", "legacy"] = "real",
) -> torch.Tensor:
    """Negative mean Pearson correlation, suitable as a loss (lower is better).

    Args:
        pred, target: shape ``(B, H, W)`` (or broadcastable).
        variant:
            - ``"real"``  : ``real(<a,b>) / sqrt(<a,a><b,b>)`` — canonical.
            - ``"abs"``   : ``|<a,b>| / sqrt(...)`` — sign-insensitive.
            - ``"legacy"``: ``|<a,b>| / sqrt(<|a|,|a|><|b|,|b|>)`` — legacy.
    """
    num = _hermitian_variance(pred, target)
    if variant == "real":
        num = num.real
    elif variant in ("abs", "legacy"):
        num = num.abs()
    else:
        raise ValueError(f"Unknown pearson variant: {variant!r}")

    var_p = _hermitian_variance(pred, pred).real.clamp_min(1e-12)
    var_t = _hermitian_variance(target, target).real.clamp_min(1e-12)
    corr = num / torch.sqrt(var_p * var_t)
    return -corr.mean()


# ---------------------------------------------------------------------------
# Regularisers
# ---------------------------------------------------------------------------

def l2_power(z: torch.Tensor) -> torch.Tensor:
    """Mean squared magnitude: ``mean |z|^2``. Scalar."""
    return torch.mean(z.abs() ** 2)


def tv_cost(f: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation on the last three dims of ``f``.

    For a 3-D tensor (C, H, W) treats C as a 'z' axis; gradients are
    computed with a circular (roll) stencil — identical to the legacy
    implementation.
    """
    if f.dim() < 3:
        raise ValueError("tv_cost expects at least a 3-D tensor")
    dz = torch.roll(f, shifts=-1, dims=-3) - f
    dy = torch.roll(f, shifts=-1, dims=-2) - f
    dx = torch.roll(f, shifts=-1, dims=-1) - f
    grad_sq = dx.abs() ** 2 + dy.abs() ** 2 + dz.abs() ** 2
    return torch.sqrt(grad_sq + 1e-12).sum()


def lateral_tv_cost(f: torch.Tensor) -> torch.Tensor:
    """Lateral (x, y) total variation — no penalty across the angle axis.

    Operates on the last two dims only. Works for real or complex tensors.
    For a (A, H, W) tensor, computes spatial gradients per angle independently.
    """
    dy = torch.roll(f, shifts=-1, dims=-2) - f
    dx = torch.roll(f, shifts=-1, dims=-1) - f
    grad_sq = dx.abs() ** 2 + dy.abs() ** 2
    return torch.sqrt(grad_sq + 1e-12).mean()


def phase_entropy(z: torch.Tensor, n_bins: int = 64) -> torch.Tensor:
    """Mean Shannon entropy of the phase distribution across angles.

    For each angle slice, bins ``angle(z)`` into ``n_bins`` bins over
    ``[-pi, pi]``, computes ``H = -sum(p * log(p))``, and returns the
    mean over angles.  A clean phase map has low entropy; noisy/random
    phase approaches ``log(n_bins)``.
    """
    phases = torch.angle(z)                              # (A, H, W)
    A = phases.shape[0]
    # Normalise to [0, 1] for histogramming
    normed = (phases + torch.pi) / (2.0 * torch.pi)     # [0, 1]
    normed = normed.clamp(0.0, 1.0 - 1e-6)

    total_H = 0.0
    for i in range(A):
        counts = torch.histc(normed[i].flatten(), bins=n_bins, min=0.0, max=1.0)
        p = counts / counts.sum()
        p = p[p > 0]
        total_H += -(p * torch.log(p)).sum().item()

    return torch.tensor(total_H / A)


# ---------------------------------------------------------------------------
# Total-loss dispatcher
# ---------------------------------------------------------------------------

def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    latent_complex: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """Fidelity + regularisation as specified by ``cfg.optimisation``.

    Fidelity function (``loss_fn``):
        'MSE'     : mean squared error (legacy default)
        'Pearson' : negative Pearson correlation

    Regularisation modes (``reg_mode``):
        'none'      : no regularisation
        'L2'        : + reg_weight * mean(|U|^2)          (legacy default)
        'TV'        : + reg_weight * TV(|U|)
        'L2+TV'     : L2 + TV with the same reg_weight
        'TV_phasor' : lateral TV on exp(1j*angle(U)) — smooths phase spatially
    """
    fn = cfg.optimisation.loss_fn
    if fn == "MSE":
        fidelity = mse_complex(pred, target)
    elif fn == "Pearson":
        fidelity = pearson_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss_fn: {fn!r}")

    mode = cfg.optimisation.reg_mode
    w = cfg.optimisation.reg_weight

    if mode == "none" or w == 0.0:
        return fidelity
    if mode == "L2":
        return fidelity + w * l2_power(latent_complex)
    if mode == "TV":
        return fidelity + w * tv_cost(latent_complex.abs())
    if mode == "L2+TV":
        return fidelity + w * (l2_power(latent_complex) + tv_cost(latent_complex.abs()))
    if mode == "TV_phasor":
        phasor = torch.exp(1.0j * torch.angle(latent_complex))
        return fidelity + w * lateral_tv_cost(phasor)
    raise ValueError(f"Unknown reg_mode: {mode!r}")
