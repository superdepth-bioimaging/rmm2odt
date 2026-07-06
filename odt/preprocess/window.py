"""Separable 2D window functions. Port of apply_window.m."""
from __future__ import annotations

import torch


def apply_window(U: torch.Tensor, kind: str) -> torch.Tensor:
    """Multiply each angle's 2D field by a separable window function.

    Parameters
    ----------
    U : torch.Tensor, (Ny, Nx, N_angles), complex
    kind : {'none', 'hann', 'tukey'}

    Returns
    -------
    torch.Tensor, same shape.

    Notes
    -----
    Reproduces the windowing scaffolding in ``LT_3D_modifiedbyH_andVA.m``
    lines 98-100 (currently disabled there by ``www = 1``).
    """
    Ny, Nx = U.shape[0], U.shape[1]
    device = U.device

    kind_lower = kind.lower()
    if kind_lower == "none":
        return U

    if kind_lower == "hann":
        wy = torch.hann_window(Ny, periodic=False, dtype=torch.float32, device=device)
        wx = torch.hann_window(Nx, periodic=False, dtype=torch.float32, device=device)
    elif kind_lower == "tukey":
        wy = _tukey_window(Ny, alpha=0.5, device=device)
        wx = _tukey_window(Nx, alpha=0.5, device=device)
    else:
        raise ValueError(
            f"Unknown window kind: {kind!r}. Use 'none', 'hann', or 'tukey'."
        )

    W = (wy[:, None] * wx[None, :]).to(U.dtype)  # (Ny, Nx)
    return U * W[..., None]                      # broadcast over angle dim


def _tukey_window(n: int, alpha: float, device: torch.device) -> torch.Tensor:
    """Tukey (cosine-tapered) window of length n. Matches MATLAB tukeywin."""
    if alpha <= 0:
        return torch.ones(n, dtype=torch.float32, device=device)
    if alpha >= 1:
        return torch.hann_window(n, periodic=False, dtype=torch.float32, device=device)

    # Indices 0..n-1
    t = torch.arange(n, dtype=torch.float32, device=device) / (n - 1)
    w = torch.ones(n, dtype=torch.float32, device=device)

    # Left taper: t in [0, alpha/2]
    left = t < alpha / 2
    w[left] = 0.5 * (1 + torch.cos(torch.pi * (2 * t[left] / alpha - 1)))

    # Right taper: t in [1 - alpha/2, 1]
    right = t > 1 - alpha / 2
    w[right] = 0.5 * (1 + torch.cos(torch.pi * (2 * t[right] / alpha - 2 / alpha + 1)))

    return w
