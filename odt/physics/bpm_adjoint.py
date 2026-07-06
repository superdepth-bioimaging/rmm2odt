"""Adjoint (time-reversed) BPM for gradient computation. Port of bpm_adjoint.m."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def bpm_adjoint(
    u_forward: torch.Tensor,
    u_meas_back: torch.Tensor,
    y_field: torch.Tensor,
    PHM: torch.Tensor,
    DFR: torch.Tensor,
    DFR_Bpr: torch.Tensor,
    k0: float,
    dz: float,
    cos_factor: float = 1.0,
    s: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Time-reversed BPM: per-angle gradient contribution.

    Parameters
    ----------
    u_forward : torch.Tensor, (Ny, Nx) complex
        Field at the output plane (BPM forward result).
    u_meas_back : torch.Tensor, (Ny, Nx) complex
        Measured field backpropagated to the same plane
        (= ``ifft2(fft2(U_meas) * DFR_Mea)``).
    y_field : torch.Tensor, (Ny, Nx, Nz) complex
        Per-layer fields from ``bpm_forward``.
    PHM : torch.Tensor, (Ny, Nx, Nz) complex
        Phase modulation matrix.
    DFR, DFR_Bpr : torch.Tensor, (Ny, Nx) complex
        Forward and backpropagation kernels.
    k0, dz, cos_factor : float
        Physical scalars.
    s : torch.Tensor, (Ny, Nx, Nz) complex, optional
        Preallocated sensitivity buffer. If None, allocated here.

    Returns
    -------
    grad_contrib : torch.Tensor, (Ny, Nx, Nz) complex
        Sensitivity map for this angle.
    residual_norm_sq : float
        ``r' * r`` at the output plane (scalar, on CPU).

    Notes
    -----
    Mirrors LT_3D_modifiedbyH_andVA.m lines 232-243 verbatim.
    """
    Nz = PHM.shape[2]

    if s is None:
        s = torch.zeros_like(PHM)

    r = u_forward - u_meas_back
    # MATLAB: r(:)' * r(:) — Hermitian inner product summed to scalar.
    residual_norm_sq = torch.vdot(r.flatten(), r.flatten()).real.item()

    coef = 1j * k0 * dz / cos_factor
    for m in range(Nz - 1, -1, -1):
        Hy = torch.fft.ifft2(torch.fft.fft2(y_field[:, :, m], dim=(0, 1)) * DFR, dim=(0, 1))
        s[:, :, m] = torch.conj(coef * PHM[:, :, m]) * (torch.conj(Hy) * r)
        r = torch.fft.ifft2(
            torch.fft.fft2(torch.conj(PHM[:, :, m]) * r, dim=(0, 1)) * DFR_Bpr,
            dim=(0, 1),
        )

    return s, residual_norm_sq
