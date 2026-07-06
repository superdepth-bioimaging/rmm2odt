"""
physics.py
==========
Core optical-physics primitives.

All operations run on the last two tensor dimensions and are broadcast-safe
over leading batch / angle dimensions.

Sign / shift conventions
------------------------
*   Centred FFT:   F{x}(k) = fftshift(fft2(ifftshift(x)))
*   Centred iFFT:  F^{-1}{X}(r) = fftshift(ifft2(ifftshift(X)))

    Both expressions are correct for even AND odd N. The legacy
    `utils.py` used `ifftshift(fft2(fftshift))` which differs by a
    one-pixel phase ramp for odd N. We use the correct form.

*   Angular-spectrum propagation kernel over a distance ``d`` (same
    physical units as 1/k):

        H(kx, ky; d) = exp(-1j * K2 * d / (k + Kz))

    ``Kz = sqrt(max(k^2 - K2, 0))`` keeps the radicand non-negative;
    evanescent components are truncated (set to propagate).  The sign
    convention matches the legacy vectorised code so that positive
    ``d`` propagates in the +z direction.

*   ``d`` is a **physical** distance (micrometres, matching the units
    of 1/k). If the caller is iterating over slice indices, multiply
    by the slice pitch first.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.fft

__all__ = [
    "centered_fft2",
    "centered_ifft2",
    "angular_spectrum_kernel",
    "build_frequency_grid",
    "safe_phasor",
]


# ---------------------------------------------------------------------------
# Fourier transform helpers (operate on the last two dims)
# ---------------------------------------------------------------------------

def centered_fft2(x: torch.Tensor) -> torch.Tensor:
    """Centred 2-D FFT on the last two dimensions.

    F(x) = fftshift ∘ fft2 ∘ ifftshift (correct for even and odd N).
    Real inputs are promoted to ``complex64``.
    """
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def centered_ifft2(x: torch.Tensor) -> torch.Tensor:
    """Centred inverse 2-D FFT on the last two dimensions."""
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


# ---------------------------------------------------------------------------
# Frequency grid and propagation kernel
# ---------------------------------------------------------------------------

def build_frequency_grid(
    Nx: int,
    Ny: int,
    npix: float,
    k: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build transverse (K2) and axial (Kz) wavevector components.

    Args:
        Nx, Ny: grid size (pixels). Centred at origin.
        npix:   pixel pitch in sample space (same units as wavelength).
        k:      wavenumber in the medium, ``k = 2 pi n / lambda``.
        device: torch device for the output tensors.
        dtype:  real dtype (float32 / float64).

    Returns:
        ``(K2, Kz)`` as real tensors of shape ``(Ny, Nx)``.
        Evanescent components are clipped (``Kz = 0`` where K2 > k^2).
    """
    device = torch.device(device)
    x = torch.arange(-Nx // 2, Nx // 2, device=device, dtype=dtype)
    y = torch.arange(-Ny // 2, Ny // 2, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    Kx = (2.0 * np.pi / (Nx * npix)) * X
    Ky = (2.0 * np.pi / (Ny * npix)) * Y
    K2 = Kx * Kx + Ky * Ky

    radicand = torch.clamp(k * k - K2, min=0.0)
    Kz = torch.sqrt(radicand)
    return K2, Kz


def angular_spectrum_kernel(
    distance: torch.Tensor,
    K2: torch.Tensor,
    k: float | torch.Tensor,
    Kz: torch.Tensor,
) -> torch.Tensor:
    """Angular-spectrum propagation kernel (paraxial-form denominator).

    ``H = exp(-1j * K2 * distance / (k + Kz))``

    Broadcasts ``distance`` against ``K2 / Kz``. Typical usage::

        d = distances.view(B, 1, 1)                 # (B,1,1)
        H = angular_spectrum_kernel(d, K2, k, Kz)   # (B, Ny, Nx)

    Negative distances produce the backward-propagation kernel
    naturally (the exponent changes sign); when ``Kz`` is real this is
    the complex conjugate of forward propagation, consistent with the
    legacy ``DFR_z(dis<0) = conj(DFR_z(|dis|))`` branch.
    """
    device = K2.device
    k_t = torch.as_tensor(k, dtype=torch.complex64, device=device)
    K2c = K2.to(torch.complex64)
    Kzc = Kz.to(torch.complex64)
    d = distance.to(torch.complex64) if torch.is_tensor(distance) else torch.as_tensor(
        distance, dtype=torch.complex64, device=device
    )

    denom = k_t + Kzc
    return torch.exp(-1.0j * K2c * d / denom)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def safe_phasor(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return ``z / (|z| + eps)`` — a unit phasor of ``z``.

    Equivalent to ``exp(1j * angle(z))`` when ``|z| >> eps``, but
    numerically stable where ``|z|`` approaches zero. Preferred over
    ``torch.angle`` followed by ``torch.exp(1j * .)`` for autograd.
    """
    return z / (z.abs() + eps)
