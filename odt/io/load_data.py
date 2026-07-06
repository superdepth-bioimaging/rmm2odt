"""Load illumination/output field stacks from MATLAB .mat files.

Port of ``load_data.m`` with auto-detection of the two known variable
naming conventions used by this project.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import scipy.io


def load_data(
    data_path: str,
    variant: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load (UI_stack, U_stack) from a MATLAB .mat file.

    Parameters
    ----------
    data_path : str
        Absolute path to the .mat file.
    variant : str
        ``'illumination_output'`` -- expects ``Illumination_stack`` and
            ``Output_stack`` stored as ``(N_angles, Ny, Nx)``; permuted to
            ``(Ny, Nx, N_angles)`` in the return.
        ``'mie'`` -- expects ``UI_stack_Mie`` and ``U_stack_Mie`` already
            in ``(Ny, Nx, N_angles)`` layout.
        ``'auto'`` (default) -- try ``illumination_output`` first, then
            ``mie``.

    Returns
    -------
    UI_stack, U_stack : np.ndarray, (Ny, Nx, N_angles), complex
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file does not exist: {data_path}")

    # Use scipy.io for v7 and earlier; fall back to h5py for v7.3 (HDF5-based).
    try:
        data = scipy.io.loadmat(data_path)
        is_v73 = False
    except (NotImplementedError, ValueError):
        import h5py
        data = {}
        with h5py.File(data_path, "r") as f:
            for key in f.keys():
                arr = f[key][...]
                # MATLAB v7.3 stores complex as compound dtype with 'real' & 'imag' fields
                if arr.dtype.names and 'real' in arr.dtype.names and 'imag' in arr.dtype.names:
                    arr = arr['real'] + 1j * arr['imag']
                # h5py reads in C-order which transposes vs MATLAB's column-major;
                # transpose to get back to MATLAB's logical layout.
                if arr.ndim >= 2:
                    arr = np.transpose(arr, list(range(arr.ndim - 1, -1, -1)))
                data[key] = arr
        is_v73 = True

    has_illumi = "Illumination_stack" in data and "Output_stack" in data
    has_mie    = "UI_stack_Mie"       in data and "U_stack_Mie"   in data

    if variant == "auto":
        if has_illumi:
            variant = "illumination_output"
        elif has_mie:
            variant = "mie"
        else:
            available = ", ".join(k for k in data.keys() if not k.startswith("__"))
            raise ValueError(
                f"Could not find any known variable pair in {data_path}.\n"
                f"Available variables: {available}"
            )

    if variant == "illumination_output":
        # MATLAB:  permute(Illumination_stack, [2, 3, 1])
        # In MATLAB this is the (N_angles, Ny, Nx) -> (Ny, Nx, N_angles) reorder.
        # In numpy (after scipy.io read which preserves MATLAB layout):
        UI_stack = np.transpose(data["Illumination_stack"], (1, 2, 0))
        U_stack  = np.transpose(data["Output_stack"],       (1, 2, 0))
    elif variant == "mie":
        UI_stack = data["UI_stack_Mie"]
        U_stack  = data["U_stack_Mie"]
    else:
        raise ValueError(f"Unknown data variant: {variant!r}")

    # Ensure complex64 for memory + GPU efficiency
    UI_stack = UI_stack.astype(np.complex64)
    U_stack  = U_stack .astype(np.complex64)

    return UI_stack, U_stack
