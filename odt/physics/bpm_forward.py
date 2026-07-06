"""Beam Propagation Method (forward pass). Port of bpm_forward.m."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def bpm_forward(
    u_inc: torch.Tensor,
    PHM: torch.Tensor,
    DFR: torch.Tensor,
    y_field: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward beam propagation through a 3D phase object.

    Parameters
    ----------
    u_inc : torch.Tensor, shape (Ny, Nx), complex
        Incident field at z = 0.
    PHM : torch.Tensor, shape (Ny, Nx, Nz), complex
        Per-layer phase modulation matrix
        (typically ``exp(1j * k0 * dz * x / cos_factor)``).
    DFR : torch.Tensor, shape (Ny, Nx), complex
        Single-step Fresnel propagation kernel (in Fourier domain).
    y_field : torch.Tensor, shape (Ny, Nx, Nz), complex, optional
        Preallocated buffer for layer-wise field history. If None, allocated
        here. Caller can supply for memory efficiency on tight GPUs.

    Returns
    -------
    u_out : torch.Tensor, shape (Ny, Nx), complex
        Field after the last layer (just past z = Nz * dz).
    y_field : torch.Tensor, shape (Ny, Nx, Nz), complex
        Field at the input plane of each layer (used by ``bpm_adjoint``).

    Notes
    -----
    Ports the inner z-loop of LT_3D_modifiedbyH_andVA.m lines 226-229.
    """
    Nz = PHM.shape[2]

    if y_field is None:
        y_field = torch.zeros_like(PHM)

    u = u_inc
    for i in range(Nz):
        y_field[:, :, i] = u
        # MATLAB: ifft2(fft2(u) .* DFR) .* PHM(:,:,i)
        u = torch.fft.ifft2(torch.fft.fft2(u, dim=(0, 1)) * DFR, dim=(0, 1)) * PHM[:, :, i]
    return u, y_field
