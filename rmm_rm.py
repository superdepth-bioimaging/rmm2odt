"""Reflection-matrix toolchain (numpy-only, pruned from workspace ``utils/``).

Only what fig3d's propagated-rrmat comparison rows need:
  symmetrize_rm  -- centre-crop a non-square RM to square (utils/rm_crop.py)
  rm_to_xk       -- real-space input -> k-space input transform
  propage_rrmat  -- translate both planes of a real-real RM by axial distance d
  img_arr_to_rm  -- image stack -> RM column layout (for .npy label inputs)

Conventions (order='F' column-major, centered FFT, e^{+i k.r}) match
``ND_MST/ndmst/matrix/rm_fft.py`` and the workspace utils. CuPy dispatch is
dropped (numpy only); everything else is a faithful port.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _N_from_axis(n_axis: int) -> int:
    N = int(round(math.sqrt(n_axis)))
    if N * N != n_axis:
        raise ValueError(f"axis size {n_axis} is not a perfect square")
    return N


# ---------------------------------------------------------------------------
# symmetrize_rm  (utils/rm_crop.py)
# ---------------------------------------------------------------------------

@dataclass
class SymmetrizeInfo:
    original_shape: Tuple[int, int]
    n_dect: int
    n_ill: int
    n_kept: int
    cropped_axis: str  # 'detection' | 'illumination' | 'none'

    def __bool__(self) -> bool:
        return self.cropped_axis != "none"

    def summary(self) -> str:
        if not self:
            return f"RM symmetric ({self.n_dect}^2 x {self.n_ill}^2), no crop"
        return (f"RM was {self.original_shape}; cropped {self.cropped_axis} side "
                f"from {max(self.n_dect, self.n_ill)} to {self.n_kept} "
                f"-> ({self.n_kept**2}, {self.n_kept**2})")


def symmetrize_rm(rm, *, order: str = "F"):
    """Make an asymmetric (N_dect**2, N_ill**2) RM square by centre-cropping
    the larger side to N = min(N_dect, N_ill)."""
    if rm.ndim != 2:
        raise ValueError(f"expected 2-D RM; got shape {rm.shape}")
    rows, cols = rm.shape
    n_dect = _N_from_axis(rows)
    n_ill = _N_from_axis(cols)
    if n_dect == n_ill:
        return rm, SymmetrizeInfo((rows, cols), n_dect, n_ill, n_dect, "none")
    n_keep = min(n_dect, n_ill)
    if n_dect > n_ill:
        pre = (n_dect - n_keep) // 2
        rm4 = rm.reshape(n_dect, n_dect, n_ill * n_ill, order=order)
        rm4 = np.ascontiguousarray(rm4[pre:pre + n_keep, pre:pre + n_keep, :])
        rm_sq = rm4.reshape(n_keep * n_keep, n_ill * n_ill, order=order)
        cropped = "detection"
    else:
        pre = (n_ill - n_keep) // 2
        rm4 = rm.reshape(n_dect * n_dect, n_ill, n_ill, order=order)
        rm4 = np.ascontiguousarray(rm4[:, pre:pre + n_keep, pre:pre + n_keep])
        rm_sq = rm4.reshape(n_dect * n_dect, n_keep * n_keep, order=order)
        cropped = "illumination"
    return rm_sq, SymmetrizeInfo((rows, cols), n_dect, n_ill, n_keep, cropped)


# ---------------------------------------------------------------------------
# rm_to_xk / rm_to_kspace / rm_to_real  (utils/rm_transforms.py)
# ---------------------------------------------------------------------------

def rm_to_xk(rm, Nx_in: Optional[int] = None, Ny_in: Optional[int] = None,
             order: str = "F"):
    """Real-space input -> k-space input (output basis unchanged)."""
    n_in = rm.shape[1]
    if Nx_in is None and Ny_in is None:
        Nx_in = _N_from_axis(n_in); Ny_in = Nx_in
    elif Nx_in is None:
        Nx_in = n_in // Ny_in
    elif Ny_in is None:
        Ny_in = n_in // Nx_in
    if Nx_in * Ny_in != n_in:
        raise ValueError(f"input axis {n_in} != Ny_in*Nx_in = {Ny_in * Nx_in}")
    N_out_sq = rm.shape[0]
    M = rm.reshape(N_out_sq, Ny_in, Nx_in, order=order)
    M = np.fft.fftshift(M, axes=(1, 2))
    M = np.fft.ifft2(M, axes=(1, 2))
    M = np.fft.fftshift(M, axes=(1, 2))
    return M.reshape(N_out_sq, Ny_in * Nx_in, order=order)


def rm_to_kspace(rm, Nx=None, Ny=None, order: str = "F"):
    """Real-real RM -> k-k RM. Forward FFT on output, inverse FFT on input."""
    n = rm.shape[0]
    if Nx is None and Ny is None:
        Nx = _N_from_axis(n); Ny = Nx
    elif Nx is None:
        Nx = n // Ny
    elif Ny is None:
        Ny = n // Nx
    if Nx * Ny != n or rm.shape[1] != Nx * Ny:
        raise ValueError(f"rm shape {rm.shape} != Nx*Ny = {Nx * Ny}")
    M = rm.reshape(Ny, Nx, -1, order=order)
    M = np.fft.fftshift(M, axes=(0, 1))
    M = np.fft.fft2(M, axes=(0, 1))
    M = np.fft.fftshift(M, axes=(0, 1))
    M = np.ascontiguousarray(M.reshape(Ny * Nx, -1, order=order).T)
    M = M.reshape(Ny, Nx, -1, order=order)
    M = np.fft.fftshift(M, axes=(0, 1))
    M = np.fft.ifft2(M, axes=(0, 1))
    M = np.fft.fftshift(M, axes=(0, 1))
    return np.ascontiguousarray(M.reshape(Ny * Nx, -1, order=order).T)


def rm_to_real(rm_k, Nx=None, Ny=None, order: str = "F"):
    """Inverse of rm_to_kspace. IFFT on output, FFT on input."""
    n = rm_k.shape[0]
    if Nx is None and Ny is None:
        Nx = _N_from_axis(n); Ny = Nx
    elif Nx is None:
        Nx = n // Ny
    elif Ny is None:
        Ny = n // Nx
    if Nx * Ny != n or rm_k.shape[1] != Nx * Ny:
        raise ValueError(f"rm_k shape {rm_k.shape} != Nx*Ny = {Nx * Ny}")
    M = rm_k.reshape(Ny, Nx, -1, order=order)
    M = np.fft.ifftshift(M, axes=(0, 1))
    M = np.fft.ifft2(M, axes=(0, 1))
    M = np.fft.ifftshift(M, axes=(0, 1))
    M = np.ascontiguousarray(M.reshape(Ny * Nx, -1, order=order).T)
    M = M.reshape(Ny, Nx, -1, order=order)
    M = np.fft.ifftshift(M, axes=(0, 1))
    M = np.fft.fft2(M, axes=(0, 1))
    M = np.fft.ifftshift(M, axes=(0, 1))
    return np.ascontiguousarray(M.reshape(Ny * Nx, -1, order=order).T)


# ---------------------------------------------------------------------------
# propage_rrmat  (utils/rm_transforms.py, fft backend only)
# ---------------------------------------------------------------------------

def _build_asm_H(N: int, d: float, lmb: float, n_med: float, res: float, NA):
    lmb_eff = lmb / n_med
    df = 1.0 / (N * res)
    coords = (np.arange(N) - N // 2) * df
    fxx, fyy = np.meshgrid(coords, coords)
    f_r2 = fxx * fxx + fyy * fyy
    inv_lmb_sq = 1.0 / lmb_eff ** 2
    propagating = f_r2 < inv_lmb_sq
    if NA is not None:
        propagating = propagating & (f_r2 < (NA / lmb_eff) ** 2)
    sq = np.maximum(inv_lmb_sq - f_r2, 0.0)
    return (propagating.astype(np.float32)
            * np.exp(1j * 2 * np.pi * d * np.sqrt(sq))).astype(np.complex64)


def propage_rrmat(rm, *, lmb: float, n: float, d: float, res: float,
                  NA: Optional[float] = None, order: str = "F"):
    """Translate both planes of a square real-real rrmat by distance d:
    M_d = P(+d) M_0 P(-d), via the k-space diagonal-multiply backend."""
    if rm.ndim != 2 or rm.shape[0] != rm.shape[1]:
        raise ValueError(f"expected a square 2-D matrix; got {rm.shape}")
    N = _N_from_axis(rm.shape[0])
    H = _build_asm_H(N, d, lmb, n, res, NA)
    rm_kk = rm_to_kspace(rm, Nx=N, Ny=N, order=order)
    H_flat = H.reshape(N * N, order=order)
    rm_kk = rm_kk * H_flat[:, None] * np.conj(H_flat)[None, :]
    return rm_to_real(rm_kk, Nx=N, Ny=N, order=order)


# ---------------------------------------------------------------------------
# img_arr_to_rm  (utils/rm_transforms.py)
# ---------------------------------------------------------------------------

def img_arr_to_rm(img_array, *, order: str = "F"):
    """Image stack (N, H, W) -> RM column layout (H*W, N), complex64."""
    if img_array.ndim != 3:
        raise ValueError(f"img_array must be (N, H, W); got {img_array.shape}")
    N, H, W = img_array.shape
    return img_array.reshape(N, H * W, order=order).T.astype(np.complex64, copy=False)
