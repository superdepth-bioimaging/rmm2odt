"""Angle-subset sampling for SGD-style FISTA. Port of sample_angles.m."""
from __future__ import annotations

from typing import Optional

import numpy as np


def sample_angles(
    total_ang: int,
    ang_num: int,
    iter_idx: int,
    stochastic: bool,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Select a subset of illumination angles for one FISTA iteration.

    Parameters
    ----------
    total_ang : int
        Total number of available illumination angles.
    ang_num : int
        Number of angles to draw.
    iter_idx : int
        Current iteration index (0-based in Python; the deterministic
        pattern uses ``iter_idx`` directly as the shift amount, equivalent
        to MATLAB's ``(Iter - 1)`` for 1-based ``Iter``).
    stochastic : bool
        If True, random uniform selection without replacement (matches
        MATLAB's RAND_ANG-based loop in distribution).
        If False, deterministic cyclic stride pattern shifted by ``iter_idx``.
    rng : np.random.Generator, optional
        For stochastic mode. Defaults to the global numpy RNG.

    Returns
    -------
    np.ndarray, shape (ang_num,), int64
        0-based angle indices (MATLAB's 1-based indices minus 1).

    Notes
    -----
    Deterministic pattern (matches MATLAB ``LT_3D_share.m`` line 138 with
    the conventional 0-based index conversion at the end)::

        idx_1based = mod(0 : stride : T - stride) + (Iter - 1), T) + 1

    where ``stride = floor(total_ang / ang_num)``. We return 0-based indices
    so callers can use them directly in Python slicing.
    """
    if stochastic:
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(total_ang, size=ang_num, replace=False)

    stride = total_ang // ang_num
    base = np.arange(0, total_ang - stride + 1, stride, dtype=np.int64)
    return (base + iter_idx) % total_ang
