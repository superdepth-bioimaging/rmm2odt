"""Plug-and-Play denoisers for the BPM_ODT FISTA proximal slot.

A PnP method replaces the TV proximal operator
    prox_{lambda*TV}(v) = argmin_x 0.5||x-v||^2 + lambda*TV(x)
with a generic Gaussian denoiser D_sigma(v) (BM3D / BM4D / NL-means). The
denoiser acts on the real 3-D delta-n volume given a noise std ``sigma`` (in
delta-n units) and an optional box clamp so the RI bounds are preserved.

Mirrors ``modified_Born_series/mbs/pnp.py``. BM3D/BM4D are CPU-only (block
matching) and require the ``bm3d`` / ``bm4d`` pip packages.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _box_np(x: np.ndarray, box: Optional[Tuple[float, float]]) -> np.ndarray:
    if box is None:
        return x
    lo, hi = box
    return np.clip(x, lo, hi)


def pnp_denoise(vol,
                sigma: float,
                kind: str = "bm3d",
                box: Optional[Tuple[float, float]] = None):
    """Denoise a real 3-D volume with a PnP Gaussian denoiser, then box-clamp.

    Parameters
    ----------
    vol : torch.Tensor | np.ndarray, real, shape (Ny, Nx, Nz)
    sigma : Gaussian noise std in the volume's units (delta-n).
    kind : 'bm3d' (slice-wise 2-D) | 'bm4d' (true 3-D) | 'nlm'
    box : (lo, hi) clamp applied after denoising (RI delta-n bounds).
    """
    is_t = torch.is_tensor(vol)
    dev = vol.device if is_t else None
    arr = (vol.detach().cpu().numpy() if is_t else np.asarray(vol)).astype(np.float32)
    s = float(sigma)

    if kind == "bm3d":
        import bm3d
        out = np.stack([bm3d.bm3d(arr[..., k], sigma_psd=s)
                        for k in range(arr.shape[-1])], axis=-1)
    elif kind == "bm4d":
        import bm4d
        out = bm4d.bm4d(arr, sigma_psd=s)
    elif kind == "nlm":
        from skimage.restoration import denoise_nl_means
        out = denoise_nl_means(arr, h=s, sigma=s, patch_size=5,
                               patch_distance=6, fast_mode=True)
    else:
        raise ValueError(f"unknown PnP denoiser kind={kind!r}")

    out = _box_np(np.asarray(out, dtype=np.float32), box)
    t = torch.from_numpy(out)
    return t.to(dev) if dev is not None else t
