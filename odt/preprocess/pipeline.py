"""Composable preprocessing pipeline. Port of apply_pipeline.m."""
from __future__ import annotations

from typing import Mapping, Tuple

import torch

from odt.preprocess.normalize import normalize_intensity, normalize_global_max
from odt.preprocess.phase_ramp import remove_phase_ramp
from odt.preprocess.phase_unwrap import unwrap_phase_2d
from odt.preprocess.window import apply_window


def apply_pipeline(
    U: torch.Tensor,
    UI: torch.Tensor,
    pp: Mapping = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the full preprocessing pipeline in a fixed order:

    1. window           (apply_window on both UI and U)
    2. intensity_normalize (normalize_intensity on U only)
    3. phase_ramp_remove   (remove_phase_ramp on U using UI)
    4. phase_unwrap_2d     (unwrap_phase_2d on U)

    The ordering matters: phase-ramp removal must come AFTER intensity
    normalization (so the carrier estimation is on a properly-scaled
    field) but BEFORE 2D phase unwrap (since the ramp is the dominant
    gradient).

    Parameters
    ----------
    U, UI : torch.Tensor, (Ny, Nx, N_angles), complex
    pp : mapping, optional
        Preprocessing config. All fields optional. Schema::

            pp.window               'none' (default) | 'hann' | 'tukey'
            pp.intensity_normalize  False (default)
                                      | True / 'per_angle_mean'
                                          -> U[:,:,a] /= mean(|U[:,:,a]|)
                                      | 'global_max'
                                          -> U /= max(|U|) (whole stack);
                                             preserves per-angle relative
                                             brightness, only rescales level
            pp.phase_ramp_remove    dict or False
                .enable      bool
                .roi         (y1, y2, x1, x2) 1-based, or None for no avg
                .reapply_UI  bool (default False)
            pp.phase_unwrap_2d      False (default) | True

    Returns
    -------
    U, UI : preprocessed tensors. UI may be modified by windowing only.
    """
    if pp is None:
        pp = {}

    # 1. Window
    win = pp.get("window", "none")
    if win.lower() != "none":
        U = apply_window(U, win)
        UI = apply_window(UI, win)

    # 2. Intensity normalization
    norm = pp.get("intensity_normalize", False)
    if norm:
        if norm is True or norm == "per_angle_mean":
            # Per-angle on U only (legacy behaviour, matches MATLAB ref)
            U = normalize_intensity(U)
        elif norm == "global_max":
            # Whole-stack max|U| applied to BOTH U and UI so the U/UI ratio
            # (what BPM actually inverts) is preserved.
            U, UI = normalize_global_max(U, UI)
        else:
            raise ValueError(
                f"Unknown intensity_normalize mode: {norm!r}. "
                f"Expected False, True, 'per_angle_mean', or 'global_max'."
            )

    # 3. Phase-ramp removal
    pr = pp.get("phase_ramp_remove", {"enable": False})
    if isinstance(pr, dict) and pr.get("enable", False):
        U = remove_phase_ramp(
            U, UI,
            roi=pr.get("roi"),
            reapply_UI=pr.get("reapply_UI", False),
        )

    # 4. 2D phase unwrap
    if pp.get("phase_unwrap_2d", False):
        U = unwrap_phase_2d(U)

    return U, UI
