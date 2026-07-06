"""Subtract illumination phase carrier from measurements. Port of remove_phase_ramp.m."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch


def remove_phase_ramp(
    U: torch.Tensor,
    UI: torch.Tensor,
    *,
    roi: Optional[Sequence[int]] = None,
    reapply_UI: bool = False,
) -> torch.Tensor:
    """Implement ``LT_3D_modifiedbyH_andVA.m`` lines 103-106 chain.

    ::

        U_phaseramp_removed = U / UI
        avg_phase = mean(angle(U_phaseramp_removed[roi]))
        U_phaseramp_removed *= exp(-1j * avg_phase)
        if reapply_UI: U = U_phaseramp_removed * UI    (else just the ratio)

    Parameters
    ----------
    U, UI : torch.Tensor, (Ny, Nx, N_angles), complex
    roi : sequence of 4 ints, optional
        ``(y1, y2, x1, x2)`` 1-based **inclusive** pixel indices in the
        original MATLAB convention. We convert to 0-based slicing
        internally. ``None`` skips the averaging step.
    reapply_UI : bool
        If True, multiplies the de-ramped field back by UI to recover
        scaled units. Default False (matches the original code where the
        re-apply line was commented out).

    Returns
    -------
    torch.Tensor, same shape as U.
    """
    eps_safe = 1e-12 * UI.abs().max().item()
    UI_safe = UI.clone()
    UI_safe[UI_safe.abs() < eps_safe] = eps_safe

    U_ratio = U / UI_safe

    if roi is not None and len(roi) > 0:
        if len(roi) != 4:
            raise ValueError(f"roi must be 4 ints (y1,y2,x1,x2); got {len(roi)}")
        y1, y2, x1, x2 = (int(v) for v in roi)
        # MATLAB is 1-based inclusive; Python is 0-based, slice end exclusive.
        # roi[68:80, 78:90] in MATLAB = indices 68..80 inclusive (13 px) and 78..90 inclusive (13 px).
        # In Python: U_ratio[y1-1:y2, x1-1:x2, :]
        avg_phase = torch.angle(U_ratio[y1 - 1: y2, x1 - 1: x2, :]).mean(dim=(0, 1), keepdim=True)
        U_ratio = U_ratio * torch.exp(-1j * avg_phase)

    if reapply_UI:
        return U_ratio * UI
    return U_ratio
