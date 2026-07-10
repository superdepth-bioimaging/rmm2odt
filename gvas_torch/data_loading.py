"""
data_loading.py
===============
I/O for illumination stacks and measured CASS images.

Supported formats
-----------------
*   ``.mat`` — MATLAB v5/v7 via ``scipy.io.loadmat``. Requires the
    config to specify ``illumination_mat_key``.
*   ``.npy`` — NumPy binary. Array shape is auto-detected; the
    leading axis is treated as the angle axis if it is the smallest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import Config


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------

def _load_illumination(path: Path, mat_key: str | None) -> np.ndarray:
    """Return a complex ``(A, H, W)`` numpy array from ``path``."""
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        if not mat_key:
            raise ValueError(f"illumination_mat_key required to load {path}")
        arr = loadmat(str(path))[mat_key]
    elif path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    else:
        raise ValueError(f"Unsupported illumination format: {path.suffix}")

    # Detect axis order: if the last dim is larger than the first, assume
    # MATLAB (H, W, A) layout and transpose to (A, H, W).
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Illumination must be 3-D, got shape {arr.shape}")
    if arr.shape[-1] > arr.shape[0]:
        arr = np.transpose(arr, (2, 0, 1))  # (H, W, A) -> (A, H, W)

    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)
    else:
        arr = arr.astype(np.complex64)
    return arr


def _load_cass(path: Path, scale: float) -> np.ndarray:
    """Load measured CASS images as complex64 ``(Z, H, W)``."""
    arr = np.load(str(path))
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"CASS array must be 3-D, got shape {arr.shape}")
    arr = arr.astype(np.complex64) * np.complex64(scale)
    return arr


# ---------------------------------------------------------------------------
# Padding / extension
# ---------------------------------------------------------------------------

def _center_pad_to(ui: torch.Tensor, Ny: int, Nx: int) -> torch.Tensor:
    """Symmetric zero-pad complex ``(A, H, W)`` to ``(A, Ny, Nx)``."""
    _, H, W = ui.shape
    if (H, W) == (Ny, Nx):
        return ui
    pad_y = Ny - H
    pad_x = Nx - W
    if pad_y < 0 or pad_x < 0:
        raise ValueError(
            f"Cannot pad ({H},{W}) to ({Ny},{Nx}) — grid config is smaller than data."
        )
    py0, py1 = pad_y // 2, pad_y - pad_y // 2
    px0, px1 = pad_x // 2, pad_x - pad_x // 2
    # F.pad last dim first: (left, right, top, bottom)
    return F.pad(ui, (px0, px1, py0, py1), mode="constant", value=0.0)


def _extend_planewaves(ui: torch.Tensor, Ny: int, Nx: int) -> torch.Tensor:
    """Extend unit-amplitude plane-wave illuminations to a larger grid.

    For each angle, extracts the transverse k-vector (kx, ky) from the
    phase gradient at the centre pixel, then regenerates ``exp(i*(kx*x +
    ky*y + offset))`` on the ``(Ny, Nx)`` grid.  This preserves the exact
    k-vector and avoids the hard-edge artifact that zero-padding creates
    — especially important for large illumination angles where the
    spatial oscillation approaches the Nyquist limit of the original grid.

    Args:
        ui: ``(A, H, W)`` complex64 illumination (unit-amplitude plane waves).
        Ny, Nx: target grid size (must be >= H, W).

    Returns:
        ``(A, Ny, Nx)`` complex64 — plane waves on the larger grid.
    """
    A, H, W = ui.shape
    device = ui.device
    cy, cx = H // 2, W // 2

    # Phase step per pixel for each angle — robust to wrapping
    kx = torch.angle(ui[:, cy, cx + 1] * ui[:, cy, cx].conj())   # (A,)
    ky = torch.angle(ui[:, cy + 1, cx] * ui[:, cy, cx].conj())   # (A,)
    offset = torch.angle(ui[:, cy, cx])                           # (A,)

    # Coordinate grids centred at origin
    x = torch.arange(-Nx // 2, Nx // 2, device=device, dtype=torch.float32)
    y = torch.arange(-Ny // 2, Ny // 2, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing="ij")  # (Ny, Nx)

    # Vectorised: (A,1,1) * (1,Ny,Nx) -> (A,Ny,Nx)
    phase = (kx[:, None, None] * X[None, :, :]
             + ky[:, None, None] * Y[None, :, :]
             + offset[:, None, None])
    return torch.exp(1.0j * phase)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(
    cfg: Config,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load illumination stack, CASS images, and distance tensor.

    Returns:
        UI_stack:  (A, Ny, Nx) complex64 on ``device``
        CASS:      (Z, crop, crop) complex64 on ``device``
        distances: (Z,) float32 physical distance in micrometres on ``device``
    """
    device = torch.device(device)
    d = cfg.data

    # Illumination
    ui_np = _load_illumination(Path(d.illumination_file), d.illumination_mat_key)
    ui = torch.from_numpy(ui_np).to(device=device, dtype=torch.complex64)
    if d.extend_illumination:
        ui = _extend_planewaves(ui, cfg.grid.Ny, cfg.grid.Nx)
    elif d.pad_to_full_grid:
        ui = _center_pad_to(ui, cfg.grid.Ny, cfg.grid.Nx)
    elif ui.shape[-2:] != (cfg.grid.Ny, cfg.grid.Nx):
        raise ValueError(
            f"pad_to_full_grid=False but illumination shape {tuple(ui.shape)} "
            f"does not match grid ({cfg.grid.Ny},{cfg.grid.Nx})."
        )

    # CASS images — load, crop, normalize by max per image, then scale by n_angles
    n_angles = ui.shape[0]
    cass_np = _load_cass(Path(d.cass_file), scale=1.0)

    # Drop the first N depth frames (file order, before reverse/crop/normalise).
    # depths/distances are sliced to match below.
    skip = max(0, int(d.depth_skip_first))
    if skip:
        if skip >= cass_np.shape[0]:
            raise ValueError(
                f"depth_skip_first={skip} >= CASS frame count {cass_np.shape[0]}"
            )
        cass_np = cass_np[skip:]

    # Optional reverse of z-axis (matches legacy `CASS_img_arr[::-1]`).
    if d.reverse_cass_z:
        cass_np = cass_np[::-1].copy()

    # Optional spatial crop [y0, y1, x0, x1]
    if d.cass_crop is not None:
        y0, y1, x0, x1 = d.cass_crop
        cass_np = cass_np[:, y0:y1, x0:x1].copy()

    # Normalize each depth image by its max amplitude, then scale by n_angles
    img_max = np.abs(cass_np).max(axis=(-2, -1), keepdims=True)
    img_max = np.where(img_max < 1e-12, 1.0, img_max)
    cass_np = cass_np / img_max * n_angles

    cass = torch.from_numpy(cass_np).to(device=device, dtype=torch.complex64)

    # Match CASS to crop_size: pad up if smaller (e.g. 160 -> 200 when extending
    # illumination), centre-crop down if larger (e.g. a 320 depth-scan stack ->
    # 200). The model predicts CASS at crop_size, so the target must equal it.
    crop = cfg.grid.resolved_crop()
    _, ch, cw = cass.shape
    if ch < crop or cw < crop:
        py = max(0, crop - ch)
        px = max(0, crop - cw)
        py0, py1 = py // 2, py - py // 2
        px0, px1 = px // 2, px - px // 2
        cass = F.pad(cass, (px0, px1, py0, py1), mode="constant", value=0.0)
        _, ch, cw = cass.shape
    if ch > crop or cw > crop:
        y0 = (ch - crop) // 2
        x0 = (cw - crop) // 2
        cass = cass[:, y0:y0 + crop, x0:x0 + crop].contiguous()

    # Distances — either from a depths file or from distance_range indices.
    if d.depths_file:
        depths = np.load(str(Path(d.depths_file))).astype(np.float32)
        if skip:
            depths = depths[skip:]
        center = d.depths_center if d.depths_center is not None else 0.0
        idx = depths - center
        if len(idx) != cass.shape[0]:
            raise ValueError(
                f"depths_file has {len(idx)} entries but CASS has {cass.shape[0]}"
            )
        distances = torch.tensor(idx * cfg.optics.scan_step, dtype=torch.complex64, device=device)
    else:
        lo, hi = d.distance_range
        idx = np.arange(lo, hi, d.distance_step)
        if skip:
            idx = idx[skip:]
        if len(idx) != cass.shape[0]:
            raise ValueError(
                f"distance_range=({lo},{hi},step={d.distance_step}) yields {len(idx)} slices "
                f"but CASS has {cass.shape[0]}"
            )
        distances = torch.tensor(idx * cfg.optics.scan_step, dtype=torch.complex64, device=device)

    return ui, cass, distances


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class CASSDataset(Dataset):
    """Minimal dataset yielding ``(cass_image, distance)`` pairs."""

    def __init__(self, cass: torch.Tensor, distances: torch.Tensor):
        assert cass.shape[0] == distances.shape[0]
        self.cass = cass
        self.distances = distances

    def __len__(self) -> int:
        return self.distances.shape[0]

    def __getitem__(self, idx: int):
        return self.cass[idx], self.distances[idx]
