"""
multilayer_torch/physics.py
----------------------------
Wave-optics primitives for the multi-layer scattering model.

Split into two layers:

PyTorch ops  — run inside forward() every batch, must be fast:
    fft2c, ifft2c, change_size, propagate

Numpy helpers — called once at model __init__ to precompute buffers,
                never called inside forward():
    np_pupil_mask, np_propagation_filter, generate_layer_params_v2
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PyTorch ops  (batch-compatible, operate on last two spatial dims)
# ---------------------------------------------------------------------------

def fft2c(u: torch.Tensor) -> torch.Tensor:
    """Centred FFT2: fftshift ∘ fft2 ∘ ifftshift over last two dims."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(u, dim=(-2, -1))),
        dim=(-2, -1),
    )


def ifft2c(u: torch.Tensor) -> torch.Tensor:
    """Centred IFFT2: fftshift ∘ ifft2 ∘ ifftshift over last two dims."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(u, dim=(-2, -1))),
        dim=(-2, -1),
    )


def change_size(u: torch.Tensor, target_size: int) -> torch.Tensor:
    """Centre-crop or centre-pad last two spatial dims to target_size × target_size.

    Padding convention matches bioimaging_subfunctions.pad:
        extra pixel goes on the left/top side.

    Parameters
    ----------
    u : Tensor[..., H, W]  — complex or real
    target_size : int

    Returns
    -------
    Tensor[..., target_size, target_size]
    """
    m = u.shape[-1]
    if m == target_size:
        return u
    if m > target_size:
        n = (m - target_size) // 2
        return u[..., n:n + target_size, n:n + target_size]
    # zero-pad: extra pixel on left/top to match numpy convention
    pad = (target_size - m) // 2
    extra = (target_size - m) % 2
    # F.pad order: (left, right, top, bottom) for last two dims
    return F.pad(u, (pad + extra, pad, pad + extra, pad))


def propagate(u: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Angular Spectrum Method (ASM) propagation step.

    Parameters
    ----------
    u : Tensor[..., H, W]  complex64  — input field
    H : Tensor[H, W]       complex64  — precomputed transfer function
        H = k_mask * exp(i·2π·z·sqrt(max(1/λ² − kx² − ky², 0)))

    Returns
    -------
    Tensor[..., H, W]  complex64
    """
    return ifft2c(fft2c(u) * H)


# ---------------------------------------------------------------------------
# Numpy helpers  (called once at model __init__, never inside forward)
# ---------------------------------------------------------------------------

def np_pupil_mask(size: int, lmb: float, L: float, NA: float) -> np.ndarray:
    """Binary spatial-frequency pupil mask (kin_mask).

    Matches generate_pupil_mask in bioimaging_subfunctions exactly:
        kmask_radius = ceil(NA * k0 / dk)
                     = ceil(NA * L / lmb)   [in pixel units]
        kin_mask = sqrt(kxx² + kyy²) < kmask_radius

    Parameters
    ----------
    size : int    — field size in pixels (square)
    lmb  : float  — effective wavelength λ/n (same units as L)
    L    : float  — field-of-view length
    NA   : float  — numerical aperture

    Returns
    -------
    float32 ndarray [size, size]
    """
    kx = np.arange(-size // 2, size // 2, 1)
    ky = np.arange(-size // 2, size // 2, 1)
    kxx, kyy = np.meshgrid(kx, ky)
    kmask_radius = np.ceil(NA * L / lmb)
    kin_mask = (np.sqrt(kxx ** 2 + kyy ** 2) < kmask_radius).astype(np.float32)
    return kin_mask


def np_propagation_filter(
    size: int,
    z: float,
    lmb: float,
    L: float,
    NA: float,
) -> np.ndarray:
    """ASM transfer function H for propagation distance z.

    H[kx, ky] = k_mask · exp(i · 2π · z · sqrt(max(1/λ² − kx² − ky², 0)))

    Evanescent components (kx² + ky² > 1/λ²) are clamped to 0 under the
    square root — no exponential blow-up for large |z|.

    Frequency coordinates match generate_angular_spectrum_prop_phase_filter:
        df = 1/L,  f_i = (i − size//2) · df

    Parameters
    ----------
    size : int    — field size in pixels
    z    : float  — propagation distance (negative = away from detector)
    lmb  : float  — effective wavelength λ/n
    L    : float  — field-of-view length
    NA   : float  — numerical aperture (defines k_mask radius)

    Returns
    -------
    complex64 ndarray [size, size]
    """
    df = 1.0 / L
    coords = (np.arange(size) - size // 2) * df
    kxx, kyy = np.meshgrid(coords, coords)
    k_r2 = kxx ** 2 + kyy ** 2
    k_mask = (np.sqrt(k_r2) < NA / lmb).astype(np.float32)
    sq = np.maximum(1.0 / lmb ** 2 - k_r2, 0.0)
    H = k_mask * np.exp(1j * 2 * np.pi * z * np.sqrt(sq))
    return H.astype(np.complex64)


def generate_layer_params_v2(
    L: float,
    size: int,
    d: np.ndarray,
    NA: float,
    n: float,
) -> tuple:
    """Compute padded field sizes and extended FOV lengths for each layer.

    Direct port of bioimaging_subfunctions.generate_layer_params_v2.
    Rounds each padded size to the nearest FFT-friendly value
    (products of 2^k × {1, 3, 5, 7}) to maximise cuFFT performance.

    Parameters
    ----------
    L    : float     — base FOV length (= resolution × size)
    size : int       — base field size in pixels
    d    : ndarray   — layer axial positions in micrometers (deepest first)
    NA   : float     — numerical aperture for padding calculation
    n    : float     — medium refractive index

    Returns
    -------
    extended_Ls  : float ndarray [N_layers]   — FOV length per layer
    padded_sizes : int   ndarray [N_layers]   — padded field size per layer
    """
    temp_L = 2 * L + 2 * np.abs(d) * NA / np.sqrt(n ** 2 - NA ** 2)
    glp_padded_size = 2 * size + np.round((temp_L - 2 * L) / (2 * L) * 2 * size)

    # Round to nearest FFT-friendly size: 2^k × {1, 3, 5, 7}
    glp_generator = np.arange(1, 12)
    glp_comparator = np.zeros((11, 4), dtype=np.float32)
    glp_factorizer = np.array([1, 3, 5, 7])
    for idx, fac in enumerate(glp_factorizer):
        glp_comparator[:, idx] = (2 ** glp_generator) * fac
    glp_comparator = np.sort(glp_comparator, axis=None)

    glp_tofind_min = glp_padded_size - glp_comparator.reshape((-1, 1))
    glp_argmin = np.argmin(np.abs(glp_tofind_min), axis=0)
    glp_padded_size = glp_padded_size - np.diagonal(glp_tofind_min[glp_argmin, :])
    glp_extended_L = glp_padded_size / size * L
    return glp_extended_L, glp_padded_size.astype(int)
