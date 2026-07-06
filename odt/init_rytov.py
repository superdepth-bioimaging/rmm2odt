"""Rytov-init helper for BPM_ODT FISTA.

Runs the standalone Rytov solver from
``research_projects/rODT/modified_Born_series/rytov_standalone`` on the
loaded (UI, U) fields and returns an absolute-RI volume usable as the
FISTA starting point.

The returned volume matches the BPM_ODT shape convention ``(Ny, Nx, Nz)``
and is in natural-image lateral coords (DC at center). The caller is
expected to apply lateral ifftshift + subtract n0 before feeding the
volume into the FISTA state (matches the existing ``optim.init_path``
``init_fftshift_lateral`` / ``init_subtract_n0`` flags).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def _find_rytov_standalone() -> Path:
    """Locate ``rytov_standalone/`` via known workspace layout."""
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(6):
        cur = cur.parent
        cand = cur / "modified_Born_series" / "rytov_standalone"
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate rytov_standalone. Expected at "
        "research_projects/rODT/modified_Born_series/rytov_standalone/. "
        "Set odt.init_rytov._RYTOV_STANDALONE_PATH if it's elsewhere."
    )


def rytov_init(
    UI_stack: np.ndarray,
    U_stack: np.ndarray,
    cfg: Mapping,
) -> np.ndarray:
    """Run Rytov inversion and return ``(Ny, Nx, Nz)`` absolute-RI volume.

    Parameters
    ----------
    UI_stack, U_stack : np.ndarray, complex64, shape (Ny, Nx, Nill)
        Illumination and observation field stacks (BPM_ODT load_data
        convention — natural image space).
    cfg : Mapping
        BPM_ODT config dict. Reads ``physics``, ``geom``, and
        ``optim.rytov_init`` for solver options.

    Returns
    -------
    np.ndarray, float32, shape (Ny, Nx, Nz)
        Real part of the Rytov-reconstructed refractive index, in
        natural-image lateral coords (DC at center, not ifftshifted).
    """
    rstd = _find_rytov_standalone()
    if str(rstd) not in sys.path:
        sys.path.insert(0, str(rstd))
    from rytov import RytovSolver, BasicOpticalParameters

    rcfg = (cfg.get("optim") or {}).get("rytov_init", {}) or {}
    physics = cfg["physics"]
    geom = cfg["geom"]

    Ny, Nx, _Nill = UI_stack.shape
    if Ny != Nx:
        raise ValueError(
            f"Rytov solver requires square xy; got ({Ny}, {Nx})"
        )
    Nz = int(geom["Nz"])

    params = BasicOpticalParameters(
        size=(Ny, Nx, Nz),
        wavelength=float(physics["wl"]),
        NA=float(physics["NA"]),
        RI_bg=float(physics["n0"]),
        resolution=(float(physics["npix"]),
                    float(physics["npix"]),
                    float(physics["dz"])),
        vector_simulation=bool(rcfg.get("vector_simulation", False)),
    )

    solver = RytovSolver(
        params,
        use_non_negativity=bool(rcfg.get("use_non_negativity", False)),
        non_negativity_iter=int(rcfg.get("non_negativity_iter", 100)),
        unwrap=bool(rcfg.get("unwrap", True)),
        unwrap_method=str(rcfg.get("unwrap_method", "ls")),
    )

    # Rytov expects (Nx, Ny, npol=1, Nill); BPM_ODT load_data returns
    # (Ny, Nx, Nill) — add a singleton npol axis at position 2.
    inp_4d = UI_stack[:, :, None, :].astype(np.complex64)
    out_4d = U_stack [:, :, None, :].astype(np.complex64)
    inp_t = torch.from_numpy(np.ascontiguousarray(inp_4d))
    out_t = torch.from_numpy(np.ascontiguousarray(out_4d))

    RI_complex, _mask = solver.solve(inp_t, out_t)
    RI_real = RI_complex.detach().cpu().numpy().real.astype(np.float32)
    return RI_real
