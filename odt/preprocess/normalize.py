"""Intensity normalization for the output field stack U.

Two modes:
  * ``normalize_intensity``     — per-angle: divides each angle by its spatial
    mean modulus; preserves per-angle shape but loses per-angle relative
    brightness. Port of normalize_intensity.m / LT_3D_modifiedbyH_andVA.m.
  * ``normalize_global_max``    — whole-stack: divides every voxel of U by the
    single scalar ``max(|U|)`` across angles, y, x. Preserves per-angle
    relative brightness; only rescales the absolute level so the brightest
    voxel in the stack sits at unit amplitude.
"""
from __future__ import annotations

import torch


def normalize_intensity(U: torch.Tensor) -> torch.Tensor:
    """Per-angle: divide each angle's field by its spatial mean modulus.

    After this, ``mean(abs(U[:, :, a])) == 1`` for every angle ``a``.

    Parameters
    ----------
    U : torch.Tensor, (Ny, Nx, N_angles), complex

    Returns
    -------
    torch.Tensor, same shape.

    Notes
    -----
    Equivalent to ``LT_3D_modifiedbyH_andVA.m`` line 39:
    ``U_stack_In = U_stack_In ./ mean(abs(U_stack_In), [1, 2])``.
    Zero-intensity angles are left untouched (no division by zero).
    """
    means = U.abs().mean(dim=(0, 1), keepdim=True)
    safe_means = torch.where(means == 0, torch.ones_like(means), means)
    return U / safe_means


def normalize_global_max(
    U: torch.Tensor,
    UI: "torch.Tensor | None" = None,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """Whole-stack: divide U (and UI, if provided) by the single scalar
    ``max(|U|)`` over (y, x, angle).

    After this, ``max(|U|) == 1`` over the whole stack, per-angle relative
    intensities are preserved within each stack, and the U/UI ratio is
    preserved everywhere (essential for BPM-ODT — the data-fidelity term
    compares u_forward(UI; x) against U, both of which must be on the same
    scale).

    Parameters
    ----------
    U : torch.Tensor, (Ny, Nx, N_angles), complex
    UI : torch.Tensor or None
        If provided, the same scalar is applied to UI as well. Returns
        ``(U_norm, UI_norm)`` in that case; otherwise returns just U_norm
        (for backwards compatibility / U-only normalization, which is
        usually not what you want for BPM-ODT — see notes).

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)

    Notes
    -----
    Without applying the same scalar to UI, the BPM forward (which uses UI
    at its original scale) overshoots the measured U by a factor of
    ``max|U|``, biasing the recon's Delta_n DOWN by the same factor.
    """
    gmax = U.abs().max()
    if gmax == 0:
        return U if UI is None else (U, UI)
    U_norm = U / gmax
    if UI is None:
        return U_norm
    return U_norm, UI / gmax
