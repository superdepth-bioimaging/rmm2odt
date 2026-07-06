"""Filter illumination + measurement pairs by an NA cutoff.

Recovers the transverse wave-vector of each illumination via 2D FFT
argmax and converts it to a physical NA value. Useful for ablations
that ask "what does the reconstruction look like if we only use
angles within NA <= 0.6?".
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def angle_na_from_illumination(
    UI: np.ndarray,
    wl: float,
    npix: float,
) -> np.ndarray:
    """Estimate the NA of each plane-wave illumination via 2D FFT argmax.

    Parameters
    ----------
    UI : np.ndarray, shape (Ny, Nx, N_angles), complex
        Illumination field stack.
    wl : float
        Vacuum wavelength (same length unit as ``npix``).
    npix : float
        Lateral pixel pitch in real space.

    Returns
    -------
    NA : np.ndarray, shape (N_angles,), float
        Physical NA = n0 * sin(theta) for each angle, recovered from
        the FFT peak via NA = wl * sqrt((ky/Ny)^2 + (kx/Nx)^2) / npix.

    Notes
    -----
    Assumes each illumination is dominated by a single plane-wave
    component (clean argmax). For mixed-mode illuminations (e.g. speckle
    patterns), use a different selection strategy.
    """
    if UI.ndim != 3:
        raise ValueError(f"UI must be 3D (Ny, Nx, N_angles); got {UI.ndim}D.")

    Ny, Nx, N = UI.shape
    F = np.fft.fft2(UI, axes=(0, 1))
    mag = np.abs(F)

    flat = mag.reshape(Ny * Nx, N).argmax(axis=0)
    ky_idx = flat // Nx
    kx_idx = flat %  Nx

    # FFT bin -> signed frequency: bins >= N/2 wrap to negative
    ky_signed = np.where(ky_idx >= Ny // 2, ky_idx - Ny, ky_idx)
    kx_signed = np.where(kx_idx >= Nx // 2, kx_idx - Nx, kx_idx)

    NA = wl * np.sqrt((ky_signed / (Ny * npix)) ** 2
                      + (kx_signed / (Nx * npix)) ** 2)
    return NA.astype(np.float64)


def filter_angles_by_na(
    UI: np.ndarray,
    U:  np.ndarray,
    NA_per_angle: np.ndarray,
    *,
    max_NA: float,
    min_NA: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep angles with ``min_NA <= NA_per_angle <= max_NA``.

    Parameters
    ----------
    UI, U : np.ndarray, shape (Ny, Nx, N_angles)
        Illumination + measurement stacks (co-indexed).
    NA_per_angle : np.ndarray, shape (N_angles,)
        Output of :func:`angle_na_from_illumination`.
    max_NA : float
        Upper cutoff (inclusive).
    min_NA : float, default 0.0
        Lower cutoff (inclusive). Use > 0 for shell-only reconstructions.

    Returns
    -------
    UI_sub, U_sub : np.ndarray, shape (Ny, Nx, n_keep)
    kept_idx : np.ndarray of int64, shape (n_keep,)
        Original indices retained (0-based).
    kept_NA : np.ndarray of float64, shape (n_keep,)
        NA values of the retained angles.

    Raises
    ------
    ValueError
        If no angles satisfy the cutoff or shapes are inconsistent.
    """
    if UI.shape != U.shape:
        raise ValueError(f"UI shape {UI.shape} != U shape {U.shape}.")
    if UI.shape[2] != NA_per_angle.shape[0]:
        raise ValueError(
            f"N_angles mismatch: UI has {UI.shape[2]}, "
            f"NA_per_angle has {NA_per_angle.shape[0]}."
        )
    if min_NA > max_NA:
        raise ValueError(f"min_NA ({min_NA}) > max_NA ({max_NA}).")

    mask = (NA_per_angle >= min_NA) & (NA_per_angle <= max_NA)
    kept_idx = np.nonzero(mask)[0].astype(np.int64)
    if kept_idx.size == 0:
        raise ValueError(
            f"No angles match NA ∈ [{min_NA}, {max_NA}]. "
            f"Full NA range present: "
            f"[{NA_per_angle.min():.3f}, {NA_per_angle.max():.3f}]."
        )

    UI_sub = UI[:, :, kept_idx].copy()
    U_sub  = U [:, :, kept_idx].copy()
    kept_NA = NA_per_angle[kept_idx]
    return UI_sub, U_sub, kept_idx, kept_NA
