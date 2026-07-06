"""3D anisotropic Total Variation cost. Port of TV_cost.m / tv_cost.m."""
from __future__ import annotations

import torch


def tv_cost(f: torch.Tensor) -> float:
    """L1-of-gradient sum of a 3D volume with circular boundary conditions.

    Parameters
    ----------
    f : torch.Tensor, (Ny, Nx, Nz), real or complex

    Returns
    -------
    float
        ``sum_{i,j,k} sqrt(|fx|^2 + |fy|^2 + |fz|^2)``
        with forward differences and wrap-around at edges.

    Notes
    -----
    Mirrors the MATLAB ``TV_cost.m`` exactly. Used for cost reporting only.
    """
    # Forward differences with circular boundary, equivalent to
    # MATLAB:  shift(1:end-1,:,:) = f(2:end,:,:); shift(end,:,:) = f(1,:,:);
    fx = torch.roll(f, shifts=-1, dims=0) - f
    fy = torch.roll(f, shifts=-1, dims=1) - f
    fz = torch.roll(f, shifts=-1, dims=2) - f

    grad_mag = torch.sqrt(
        (fx * torch.conj(fx)).real
        + (fy * torch.conj(fy)).real
        + (fz * torch.conj(fz)).real
    )
    return grad_mag.sum().item()
