"""
multilayer_torch/data_loading.py
---------------------------------
R-matrix data loading utilities.

Provides an alternative data loading path: instead of separate illumination
and labels .npy files, load a single HDF5 .mat file containing a reflection
matrix and generate illumination/labels programmatically.

Public API
----------
    load_rrmat(path, wl, n0, dx)
        Load reflection matrix from HDF5 .mat file.

    rrmat_to_illum_labels(rr, Nx, sigma)
        Convert R-matrix to illumination and labels arrays.

    estimate_rod(rr, Nx, threshold_ratio)
        Auto-detect radius-of-detection from center source response.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Keys to search for the reflection matrix in HDF5 files
_RRMAT_KEYS = ['rr_mat', 'rrmat', 'full_rr_mat', 'Rk', 'R']


def load_rrmat(
    path: str | Path,
    wl: float | None = None,
    n0: float | None = None,
    dx: float | None = None,
) -> tuple[np.ndarray, int, int, float, float, float]:
    """Load reflection matrix from an HDF5 .mat file.

    Searches for the R-matrix using common variable names. Optical parameters
    (wavelength, refractive index, pixel pitch) are read from the file if
    present, with function arguments taking precedence as overrides.

    Supports both square (Nx_det == Nx_illum) and non-square R-matrices
    where illumination and detection grids have different sizes.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 .mat file.
    wl : float, optional
        Wavelength in micrometers. Overrides file value if provided.
    n0 : float, optional
        Medium refractive index. Overrides file value if provided.
    dx : float, optional
        Pixel pitch (um/pixel). Overrides file value if provided.

    Returns
    -------
    rr : ndarray [Nx_det*Nx_det, Nx_illum*Nx_illum] complex64
        Reflection matrix. Rows = detectors, columns = sources.
    Nx_det : int
        Detection grid dimension (pixels per side) -- determines labels_size.
    Nx_illum : int
        Illumination grid dimension (pixels per side) -- determines illumination_size.
    wl : float
        Wavelength (um).
    n0 : float
        Medium refractive index.
    dx : float
        Pixel pitch (um/pixel).
    """
    import h5py

    path = Path(path)
    logger.info(f'Loading R-matrix from {path}')

    with h5py.File(str(path), 'r') as f:
        # Find the reflection matrix variable
        rr = None
        for key in _RRMAT_KEYS:
            if key in f:
                # Check structured (real+imag) vs already-complex without
                # materialising the full array. For very large rrmats this
                # avoids a 14 GB complex128 intermediate that otherwise
                # pushes peak RSS well past tight cgroup memory limits.
                dset_dtype = f[key].dtype
                if dset_dtype.names and 'real' in dset_dtype.names:
                    raw = np.array(f[key])
                    # Build complex64 in-place from the structured pair —
                    # no complex128 temp.
                    rr = np.empty(raw.shape, dtype=np.complex64)
                    rr.real = raw['real']
                    rr.imag = raw['imag']
                    del raw
                else:
                    rr = np.array(f[key]).astype(np.complex64, copy=False)
                # h5py transposes from MATLAB column-major; restore (N_det, N_illum).
                # Keep as a view (rr.T) — materialising would cost another 14 GB.
                rr = rr.T
                import gc; gc.collect()
                logger.info(f'  Found R-matrix key "{key}", shape {rr.shape}')
                break

        if rr is None:
            raise KeyError(
                f'No reflection matrix found in {path}. '
                f'Searched keys: {_RRMAT_KEYS}'
            )

        # Read optional scalar parameters from file
        def _read_scalar(key, default):
            if key in f:
                return float(np.array(f[key]).flatten()[0])
            return default

        wl_v = wl if wl is not None else _read_scalar('wavelength', None)
        n0_v = n0 if n0 is not None else _read_scalar('n0', None)
        dx_v = dx if dx is not None else _read_scalar('dx', None)

    # Derive grid dimensions: rows = detectors, columns = sources
    Nx_det = int(np.sqrt(rr.shape[0]))
    Nx_illum = int(np.sqrt(rr.shape[1]))
    if Nx_det * Nx_det != rr.shape[0]:
        raise ValueError(
            f'R-matrix row count {rr.shape[0]} is not a perfect square: '
            f'sqrt({rr.shape[0]}) = {np.sqrt(rr.shape[0]):.2f}'
        )
    if Nx_illum * Nx_illum != rr.shape[1]:
        raise ValueError(
            f'R-matrix column count {rr.shape[1]} is not a perfect square: '
            f'sqrt({rr.shape[1]}) = {np.sqrt(rr.shape[1]):.2f}'
        )

    # Validate all parameters are available
    if wl_v is None:
        raise ValueError('Wavelength not found in .mat file and not provided in config')
    if n0_v is None:
        raise ValueError('Refractive index (n0) not found in .mat file and not provided in config')
    if dx_v is None:
        raise ValueError('Pixel pitch (dx) not found in .mat file and not provided in config')

    if Nx_det == Nx_illum:
        logger.info(f'  Nx={Nx_det}, wl={wl_v}, n0={n0_v}, dx={dx_v}')
    else:
        logger.info(
            f'  Nx_det={Nx_det}, Nx_illum={Nx_illum} (non-square R-matrix), '
            f'wl={wl_v}, n0={n0_v}, dx={dx_v}'
        )
    return rr, Nx_det, Nx_illum, float(wl_v), float(n0_v), float(dx_v)


def rrmat_to_illum_labels(
    rr: np.ndarray,
    Nx_det: int,
    Nx_illum: int | None = None,
    sigma: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert reflection matrix to illumination and labels arrays.

    For each source point i, generates:
    - illumination[i]: delta function (or Gaussian spot) at scanning position i
      in the illumination grid (Nx_illum x Nx_illum)
    - labels[i]: R-matrix column i reshaped to the detection grid
      (Nx_det x Nx_det)

    Supports non-square R-matrices where Nx_illum != Nx_det.
    Uses MATLAB column-major indexing: index i maps to
    (row=i%Nx_illum, col=i//Nx_illum).

    Parameters
    ----------
    rr : ndarray [Nx_det*Nx_det, Nx_illum*Nx_illum] complex64
        Reflection matrix. Rows = detectors, columns = sources.
    Nx_det : int
        Detection grid dimension.
    Nx_illum : int, optional
        Illumination grid dimension. If None, assumed equal to Nx_det.
    sigma : float
        Gaussian spot width. 0 or <=0.1 means delta function.

    Returns
    -------
    illum : ndarray [N_src, Nx_illum, Nx_illum] complex64
        Illumination patterns (one per source).
    labels : ndarray [N_src, Nx_det, Nx_det] complex64
        Ground-truth reflected fields (one per source).
    """
    if Nx_illum is None:
        Nx_illum = Nx_det

    if Nx_illum > Nx_det:
        raise ValueError(
            f'Nx_illum ({Nx_illum}) > Nx_det ({Nx_det}): detection grid must be '
            f'>= illumination grid. Check your R-matrix data or config.'
        )

    N_src = Nx_illum * Nx_illum
    illum = np.zeros((N_src, Nx_illum, Nx_illum), dtype=np.complex64)
    labels = np.zeros((N_src, Nx_det, Nx_det), dtype=np.complex64)

    use_gaussian = sigma > 0.1

    if use_gaussian:
        yy, xx = np.mgrid[0:Nx_illum, 0:Nx_illum]

    for i in range(N_src):
        # MATLAB column-major indexing on the illumination grid
        row = i % Nx_illum
        col = i // Nx_illum

        if use_gaussian:
            spot = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))
            illum[i] = (spot / spot.sum() * Nx_illum).astype(np.complex64)
        else:
            illum[i, row, col] = Nx_illum  # delta function

        # Label = R-matrix column i reshaped to detection grid (MATLAB F-order)
        labels[i] = rr[:, i].reshape(Nx_det, Nx_det, order='F')

    if Nx_det == Nx_illum:
        logger.info(
            f'Generated {N_src} illumination/label pairs from R-matrix '
            f'(Nx={Nx_det}, sigma={sigma}, mode={"gaussian" if use_gaussian else "delta"})'
        )
    else:
        logger.info(
            f'Generated {N_src} illumination/label pairs from non-square R-matrix '
            f'(illum={Nx_illum}x{Nx_illum}, labels={Nx_det}x{Nx_det}, '
            f'sigma={sigma}, mode={"gaussian" if use_gaussian else "delta"})'
        )
    return illum, labels


def estimate_rod(
    rr: np.ndarray,
    Nx_det: int,
    Nx_illum: int | None = None,
    threshold_ratio: float = 0.05,
) -> int:
    """Estimate radius-of-detection from the center source response.

    Finds the radial extent where the amplitude of the center source's
    backscattered field stays above a threshold fraction of the peak.

    Parameters
    ----------
    rr : ndarray [Nx_det*Nx_det, Nx_illum*Nx_illum] complex64
        Reflection matrix.
    Nx_det : int
        Detection grid dimension.
    Nx_illum : int, optional
        Illumination grid dimension. If None, assumed equal to Nx_det.
    threshold_ratio : float
        Fraction of peak amplitude to use as detection threshold.

    Returns
    -------
    rod : int
        Estimated ROD in pixels (diameter).
    """
    if Nx_illum is None:
        Nx_illum = Nx_det

    # Center source index on the illumination grid (F-order)
    r_c_illum = Nx_illum // 2
    c_c_illum = Nx_illum // 2
    idx_center = r_c_illum + c_c_illum * Nx_illum

    # Extract center source's response on the detection grid
    label = rr[:, idx_center].reshape(Nx_det, Nx_det, order='F')
    amp = np.abs(label)
    threshold = threshold_ratio * amp.max()

    # Radial scan from edge inward on detection grid
    r_c_det = Nx_det // 2
    c_c_det = Nx_det // 2
    yy, xx = np.mgrid[0:Nx_det, 0:Nx_det]
    rr_dist = np.sqrt((yy - r_c_det) ** 2 + (xx - c_c_det) ** 2)

    max_r = Nx_det // 2
    for ri in range(max_r - 1, 0, -1):
        mask = (rr_dist >= ri) & (rr_dist < ri + 1)
        if mask.any() and np.mean(amp[mask]) > threshold:
            rod = 2 * ri + 1
            logger.info(f'Auto-detected ROD = {rod} pixels (threshold_ratio={threshold_ratio})')
            return rod

    logger.info(f'ROD auto-detection: no threshold crossing, using full grid Nx_det={Nx_det}')
    return Nx_det
