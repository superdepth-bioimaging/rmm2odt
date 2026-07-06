"""Stopping criteria for iterative ODT reconstruction.

Each function returns a **scalar metric** at the current iteration. No
thresholding inside — the caller decides when to stop based on the
trajectory of the metric over iterations.

All functions are pure (no side effects) and accept only numpy/torch
arrays, so they can be called from any solver (FISTA, ADMM, Adam, ...).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Relative reconstruction change
# ---------------------------------------------------------------------------

def relative_reconstruction_change(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
) -> float:
    """||x_curr - x_prev|| / ||x_curr||.

    Measures whether the *image* has stopped changing. Robust to SGD noise
    because the reconstruction is a running average of many gradient steps.

    Typical threshold: 1e-4 for convergence.
    """
    diff_norm = torch.linalg.norm(x_curr - x_prev).item()
    curr_norm = torch.linalg.norm(x_curr).item()
    if curr_norm < 1e-30:
        return float("inf")
    return diff_norm / curr_norm


# ---------------------------------------------------------------------------
# 2. Morozov discrepancy principle
# ---------------------------------------------------------------------------

def morozov_discrepancy(
    cost_d: float,
    n_measurements: int,
    noise_sigma: float,
) -> float:
    """Ratio of data-fidelity cost to the expected noise energy.

    Returns ``cost_d / (0.5 * n_measurements * sigma^2)``.

    * ratio >> 1 → under-fitting (model hasn't explained the data yet)
    * ratio ≈  1 → optimal (residual matches noise level)
    * ratio << 1 → over-fitting (fitting noise)

    Parameters
    ----------
    cost_d : float
        Current data-fidelity term: ``0.5 / ang_num * sum(|r|^2)``.
    n_measurements : int
        Total number of scalar measurements = ``ang_num * Ny * Nx``.
    noise_sigma : float
        Estimated per-measurement noise standard deviation.
    """
    expected = 0.5 * n_measurements * noise_sigma ** 2
    if expected < 1e-30:
        return float("inf")
    return cost_d / expected


def estimate_noise_sigma(U_stack: torch.Tensor, corner_frac: float = 0.1) -> float:
    """Estimate measurement noise sigma from corner regions of U_stack.

    Uses the four corners of each angle's 2D field (where there is
    typically no sample) to estimate the standard deviation of the
    complex-valued noise.

    Parameters
    ----------
    U_stack : (Ny, Nx, N_angles) complex tensor
    corner_frac : fraction of each side used for the corner crop

    Returns
    -------
    float : estimated sigma (real-valued noise std)
    """
    Ny, Nx = U_stack.shape[0], U_stack.shape[1]
    cy = max(1, int(Ny * corner_frac))
    cx = max(1, int(Nx * corner_frac))
    corners = torch.cat([
        U_stack[:cy,  :cx,  :].reshape(-1),
        U_stack[:cy,  -cx:, :].reshape(-1),
        U_stack[-cy:, :cx,  :].reshape(-1),
        U_stack[-cy:, -cx:, :].reshape(-1),
    ])
    # Noise std of the real and imaginary parts
    sigma = corners.real.std().item()
    return sigma


# ---------------------------------------------------------------------------
# 3. Cross-validation error (held-out angles)
# ---------------------------------------------------------------------------

def cross_validation_error(
    x: torch.Tensor,
    held_out_UI: torch.Tensor,
    held_out_U: torch.Tensor,
    DFR: torch.Tensor,
    DFR_Ori: torch.Tensor,
    DFR_Mea: torch.Tensor,
    k0: float,
    dz: float,
) -> float:
    """Prediction error on held-out angles not used during reconstruction.

    Forward-propagates through ``x`` for each held-out angle and computes
    the mean-squared error between the predicted and actual measured field.

    Returns
    -------
    float : mean |u_predicted - u_measured|^2 over all held-out angles.
    """
    from odt.physics.bpm_forward import bpm_forward

    Nz = x.shape[2]
    n_held = held_out_UI.shape[2]
    PHM = torch.exp(1j * (k0 * dz) * x.to(torch.complex64))

    total_err = 0.0
    for a in range(n_held):
        u_inc = torch.fft.ifft2(
            torch.fft.fft2(held_out_UI[:, :, a], dim=(0, 1)) * DFR_Ori,
            dim=(0, 1),
        )
        u_out, _ = bpm_forward(u_inc, PHM, DFR, None)
        u_meas = torch.fft.ifft2(
            torch.fft.fft2(held_out_U[:, :, a], dim=(0, 1)) * DFR_Mea,
            dim=(0, 1),
        )
        err = (u_out - u_meas).abs().pow(2).sum().item()
        total_err += err

    n_pixels = x.shape[0] * x.shape[1]
    return total_err / (n_held * n_pixels)


# ---------------------------------------------------------------------------
# 4. Smoothed cost slope
# ---------------------------------------------------------------------------

def smoothed_cost_slope(
    cost_history: np.ndarray,
    iter_idx: int,
    window: int = 10,
) -> float:
    """Relative slope of the moving-averaged cost curve.

    Returns ``(cost_smooth[now] - cost_smooth[now - window]) / (window * cost_smooth[now])``.

    Negative = cost still decreasing; close to zero = plateau.
    Typical threshold: ``|slope| < 0.01`` per iteration.

    Parameters
    ----------
    cost_history : 1D array of cost values (up to iter_idx filled)
    iter_idx : current iteration (0-based)
    window : averaging window size
    """
    if iter_idx < window:
        return float("-inf")  # not enough data yet
    recent = cost_history[max(0, iter_idx - window + 1): iter_idx + 1].mean()
    earlier = cost_history[max(0, iter_idx - 2 * window + 1): iter_idx - window + 1].mean()
    if abs(recent) < 1e-30:
        return 0.0
    return (recent - earlier) / (window * abs(recent))


# ---------------------------------------------------------------------------
# 5. Residual entropy
# ---------------------------------------------------------------------------

def residual_entropy(
    r_stack: torch.Tensor,
    n_bins: int = 256,
) -> float:
    """Shannon entropy of the residual histogram.

    When the residual is pure Gaussian noise, entropy =
    0.5 * log2(2 * pi * e * sigma^2). When the model under-fits,
    the residual has structure → entropy is lower.

    Parameters
    ----------
    r_stack : (Ny, Nx) or (Ny, Nx, N) complex tensor
        Residual field(s). Uses the real part for the histogram.
    n_bins : int
    """
    vals = r_stack.real.detach().cpu().numpy().flatten()
    return _shannon_entropy(vals, n_bins)


# ---------------------------------------------------------------------------
# 6. Reconstruction entropy
# ---------------------------------------------------------------------------

def reconstruction_entropy(
    x: torch.Tensor,
    n_bins: int = 256,
) -> float:
    """Shannon entropy of the voxel-value histogram of the reconstruction.

    Low at start (all zeros), increases as structure emerges, plateaus
    when converged, increases again if overfitting adds noise.

    Parameters
    ----------
    x : (Ny, Nx, Nz) real tensor
    n_bins : int
    """
    vals = x.detach().cpu().numpy().flatten()
    return _shannon_entropy(vals, n_bins)


# ---------------------------------------------------------------------------
# 7. Peak SNR
# ---------------------------------------------------------------------------

def snr_peak(x: torch.Tensor, bg_percentile: float = 50.0) -> float:
    """Peak signal-to-noise ratio: max(|x|) / std(background).

    Background is estimated as voxels with |x| below the given percentile.

    Returns
    -------
    float : SNR in linear scale (not dB). Use 20*log10(snr) for dB.
    """
    x_np = x.detach().cpu().numpy().flatten()
    threshold = np.percentile(np.abs(x_np), bg_percentile)
    bg = x_np[np.abs(x_np) <= threshold]
    if len(bg) < 10:
        return 0.0
    noise_std = bg.std()
    if noise_std < 1e-30:
        return float("inf")
    return float(np.abs(x_np).max() / noise_std)


# ---------------------------------------------------------------------------
# 8. TV relative change
# ---------------------------------------------------------------------------

def tv_relative_change(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
) -> float:
    """Relative change in the Total Variation of the reconstruction.

    ``(TV(x_curr) - TV(x_prev)) / TV(x_prev)``

    Positive = TV increasing (possibly overfitting / adding spurious edges).
    Negative = TV decreasing (denoising phase).
    Near zero = TV stabilised.
    """
    from odt.reg.tv_cost import tv_cost
    tv_curr = tv_cost(x_curr)
    tv_prev = tv_cost(x_prev)
    if abs(tv_prev) < 1e-30:
        return 0.0
    return (tv_curr - tv_prev) / abs(tv_prev)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _shannon_entropy(values: np.ndarray, n_bins: int) -> float:
    """Shannon entropy of a histogram in bits."""
    hist, _ = np.histogram(values, bins=n_bins)
    hist = hist[hist > 0].astype(np.float64)
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))
