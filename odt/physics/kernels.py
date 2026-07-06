"""Fresnel diffraction kernels for the Beam Propagation Method.

Direct port of ``+odt/+physics/build_kernels.m``. All numerics preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from odt.util.backend import get_device, to_device


@dataclass
class Kernels:
    """Container for the four diffraction kernels and physical scalars.

    Shape of each kernel: ``(Ny, Nx)`` complex64.
    """
    DFR: torch.Tensor      # single-step forward propagation kernel
    DFR_Bpr: torch.Tensor  # conj(DFR) — single-step backpropagation
    DFR_Mea: torch.Tensor  # center-to-measurement plane backpropagation
    DFR_Ori: torch.Tensor  # origin-plane propagation
    k0: float
    k: float
    dz: float


def build_kernels(physics: Mapping[str, float], geom: Mapping[str, int]) -> Kernels:
    """Construct Fresnel diffraction kernels for BPM.

    Parameters
    ----------
    physics : mapping
        Required keys: ``wl``, ``n0``, ``npix``, ``dz``.
    geom : mapping
        Required keys: ``Nx``, ``Ny``, ``Nz``.

    Returns
    -------
    Kernels

    Notes
    -----
    Equivalent to ``LT_3D_modifiedbyH_andVA.m`` lines 96-114 / refactored
    ``build_kernels.m``. Uses MATLAB's
    ``meshgrid(-round((N+1)/2)+1 : N-round((N+1)/2))`` indexing exactly so
    the resulting K2 layout matches before the ifftshift.
    """
    Nx, Ny, Nz = int(geom["Nx"]), int(geom["Ny"]), int(geom["Nz"])
    wl = float(physics["wl"])
    n0 = float(physics["n0"])
    npix = float(physics["npix"])
    dz = float(physics["dz"])

    k0 = 2.0 * math.pi / wl
    k  = k0 * n0

    # MATLAB:  -round((Nx+1)/2) + 1 : Nx - round((Nx+1)/2)
    #          MATLAB's round() uses round-half-away-from-zero. For positive
    #          (N+1)/2, the value is N/2 + 1 when N is even (round(2.5)=3 etc).
    #          The Python equivalent is N // 2 + 1, which works for both
    #          even and odd N.
    def _axis(n: int) -> np.ndarray:
        half = n // 2 + 1
        return np.arange(-half + 1, n - half + 1, dtype=np.float64)

    X, Y = np.meshgrid(_axis(Nx), _axis(Ny), indexing="xy")  # both (Ny, Nx)

    Kx = (2.0 * math.pi / Nx / npix) * X
    Ky = (2.0 * math.pi / Ny / npix) * Y

    K2 = np.fft.ifftshift(Kx ** 2 + Ky ** 2)
    Kz = np.sqrt(k * k - K2 + 0j)  # complex sqrt for evanescent regime

    # Diffraction kernels (complex, single)
    # MATLAB round-half-away-from-zero for round((Nz+1)/2):
    half_nz_p1 = Nz // 2 + 1                   # = round((Nz+1)/2) for any Nz
    DFR     = np.exp(-1j * K2 * dz / (k + Kz))
    DFR_Bpr = np.conj(DFR)
    DFR_Mea = np.exp(-1j * K2 * dz * math.floor((Nz - 1) / 2) / (k + Kz))
    DFR_Ori = np.conj(np.exp(-1j * K2 * dz * half_nz_p1 / (k + Kz)))

    device = get_device()
    return Kernels(
        DFR     = to_device(DFR.astype(np.complex64),     dtype=torch.complex64, device=device),
        DFR_Bpr = to_device(DFR_Bpr.astype(np.complex64), dtype=torch.complex64, device=device),
        DFR_Mea = to_device(DFR_Mea.astype(np.complex64), dtype=torch.complex64, device=device),
        DFR_Ori = to_device(DFR_Ori.astype(np.complex64), dtype=torch.complex64, device=device),
        k0      = k0,
        k       = k,
        dz      = dz,
    )
