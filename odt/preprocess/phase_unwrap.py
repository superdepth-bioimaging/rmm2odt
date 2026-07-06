"""2D phase unwrap on each angle. Port of unwrap_phase_2d.m."""
from __future__ import annotations

import numpy as np
import torch


def unwrap_phase_2d(U: torch.Tensor) -> torch.Tensor:
    """Apply 2D phase unwrap (along dim 0 then dim 1) preserving magnitude.

    ``U_new = abs(U) * exp(1j * unwrap(unwrap(angle(U), dim=0), dim=1))``

    Parameters
    ----------
    U : torch.Tensor, (Ny, Nx, N_angles), complex

    Returns
    -------
    torch.Tensor, same shape.

    Notes
    -----
    Matches ``LT_3D_modifiedbyH_andVA.m`` line 107 / the active line in
    ``LT_3D_modifiedbyH.m``. Implemented via numpy for simplicity since
    PyTorch doesn't have a native ``unwrap`` and the cost is negligible
    relative to the FFTs in the FISTA loop.
    """
    device = U.device
    U_cpu = U.detach().cpu().numpy()
    mag = np.abs(U_cpu)
    phase = np.angle(U_cpu)
    phase = np.unwrap(np.unwrap(phase, axis=0), axis=1)
    out = mag * np.exp(1j * phase)
    return torch.from_numpy(out.astype(np.complex64)).to(device=device)
