"""Device + dtype helpers for the PyTorch backend.

We use PyTorch as the array library throughout. NumPy is used only at I/O
boundaries (loading .mat files, saving outputs, matplotlib visualization).

Conventions:
- Compute dtypes: ``torch.float32`` for real arrays, ``torch.complex64`` for
  complex arrays. This matches MATLAB ``single`` precision.
- Device: ``cuda`` if available, else ``cpu``. Selected once via
  :func:`get_device` and threaded through the rest of the code.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


_DEVICE: Optional[torch.device] = None


def get_device(prefer: str = "cuda") -> torch.device:
    """Return (and cache) the compute device.

    Parameters
    ----------
    prefer : str
        ``'cuda'`` (default) or ``'cpu'``. CUDA is used only when available.

    Returns
    -------
    torch.device
    """
    global _DEVICE
    if _DEVICE is None:
        if prefer == "cuda" and torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        else:
            _DEVICE = torch.device("cpu")
    return _DEVICE


def reset_device() -> None:
    """Forget the cached device. Useful in tests."""
    global _DEVICE
    _DEVICE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_device(array, dtype=None, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert numpy array (or torch tensor) to a tensor on the compute device.

    Parameters
    ----------
    array : numpy.ndarray | torch.Tensor
    dtype : torch.dtype, optional
        Target dtype. If None, uses ``complex64`` for complex inputs and
        ``float32`` for real inputs.
    device : torch.device, optional
        Target device. Defaults to :func:`get_device`.

    Returns
    -------
    torch.Tensor
    """
    if device is None:
        device = get_device()

    if isinstance(array, torch.Tensor):
        t = array
    else:
        t = torch.from_numpy(np.ascontiguousarray(array))

    if dtype is None:
        dtype = torch.complex64 if t.is_complex() else torch.float32

    return t.to(device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Move a tensor to CPU and convert to numpy."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def zeros(shape, dtype=torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Allocate a zero tensor on the compute device with the given dtype."""
    if device is None:
        device = get_device()
    return torch.zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    """Allocate a ones tensor on the compute device."""
    if device is None:
        device = get_device()
    return torch.ones(shape, dtype=dtype, device=device)
