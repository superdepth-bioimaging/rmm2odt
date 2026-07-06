"""
multilayer_torch/raw2rrmat.py
-----------------------------
Generate the real-space reflection matrix ``rrmat_fullroi.mat`` from raw CASS
data (a metadata ``.mat`` file + companion ``.bin`` uint16 camera frames).

Pipeline (ported from ``ESCLASS 1.4_v2/rawdata_to_rr_mat``):
    1. load_cass_raw_data    — .mat metadata + .bin uint16 camera images
    2. crop_cam_images       — centre-crop each frame to N_d x N_d
    3. gen_fft_images        — pre-shifted 2-D FFT (batched)
    4. crop_fft_images       — extract off-axis carrier sideband
    5. gen_off_axis_images   — IFFT + fftshift + horizontal flip  -> complex field
    6. gen_rrmat             — assemble reflection matrix [sz, sz, N_scan]
    7. crop_roi  (optional)  — crop a sub-region [Nx, Nx, Nx^2]
    8. save_rrmat            — write rrmat_fullroi.mat in a folder named after
                               the source raw .mat filename (next to the source).

Public API
----------
    run_raw_to_rrmat(raw_mat_path, N_d, xc, yc, Nx_roi, save)
        Execute the full pipeline and optionally save the rrmat.

    ensure_rrmat_exists(rrmat_path, raw_mat_path, ...)
        Return a path to a valid rrmat file.  If ``rrmat_path`` exists, returns
        it unchanged.  Otherwise falls back to generating one from raw data
        (requires ``raw_mat_path``).  Used by the solver to auto-generate
        rrmats on demand.
"""

from __future__ import annotations

import logging
import os
import types
from pathlib import Path

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load raw CASS data
# ---------------------------------------------------------------------------

def _struct_field(s, field):
    """Extract a scalar value from a scipy mat_struct or SimpleNamespace field."""
    val = getattr(s, field)
    if isinstance(val, np.ndarray):
        return val.flat[0]
    return val


def _dict_to_namespace(obj):
    """Recursively convert a dict (from mat73) to SimpleNamespace for dot access."""
    if isinstance(obj, dict):
        ns = types.SimpleNamespace()
        for k, v in obj.items():
            setattr(ns, k, _dict_to_namespace(v))
        return ns
    return obj


def _load_mat73(filename, variable_names):
    """Load selected variables from a MATLAB v7.3 (HDF5) .mat via mat73."""
    try:
        import mat73
    except ImportError:
        raise ImportError(
            f"'{filename}' is a MATLAB v7.3 (HDF5) file which scipy cannot read.\n"
            "Install mat73 and retry:  pip install mat73"
        )
    data = mat73.loadmat(filename)
    return {k: data[k] for k in variable_names if k in data}


def load_cass_raw_data(filename):
    """Load raw CASS data: .mat metadata + .bin camera frames.

    Supports both MATLAB <= v7 (scipy) and MATLAB v7.3 / HDF5 (mat73).

    Returns dict with keys:
        cam_images      : ndarray [N_cam, N_cam, N_scan]  uint16
        scan_params     : struct-like (tot_num_of_scan_points, ...)
        off_axis_params : struct-like (N_cam, N_K_max, kx_center, ky_center, ...)
        pco             : struct-like
        filename        : absolute source .mat path
    """
    filename = os.path.abspath(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    folder = os.path.dirname(filename)
    variable_names = ["scan_params", "off_axis_params", "pco", "RawDataFileName"]

    try:
        mat = scipy.io.loadmat(
            filename, squeeze_me=True, struct_as_record=False,
            variable_names=variable_names,
        )
        scan_params = mat["scan_params"]
        off_axis_params = mat["off_axis_params"]
        pco = mat.get("pco")
        rdfilename = mat["RawDataFileName"]
        raw_data_filename = (
            str(rdfilename.flat[0]).strip() if isinstance(rdfilename, np.ndarray)
            else str(rdfilename).strip()
        )
    except NotImplementedError:
        logger.info("  [load] MATLAB v7.3 detected, using mat73 ...")
        mat = _load_mat73(filename, variable_names)
        scan_params = _dict_to_namespace(mat.get("scan_params", {}))
        off_axis_params = _dict_to_namespace(mat.get("off_axis_params", {}))
        pco = _dict_to_namespace(mat.get("pco", {}))
        rdfilename = mat.get("RawDataFileName", "")
        raw_data_filename = str(rdfilename).strip() if rdfilename else ""

    result = {
        "scan_params": scan_params,
        "off_axis_params": off_axis_params,
        "pco": pco,
        "filename": filename,
    }

    if raw_data_filename:
        N_cam = int(_struct_field(off_axis_params, "N_cam"))
        N_scan = int(_struct_field(scan_params, "tot_num_of_scan_points"))
        bin_path = os.path.join(folder, raw_data_filename)
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(
                f"Binary data file not found: {bin_path}\n"
                f"Expected alongside the .mat file at: {folder}"
            )
        # MATLAB writes column-major; reshape with Fortran order.
        raw = np.fromfile(bin_path, dtype=np.uint16)
        result["cam_images"] = raw.reshape((N_cam, N_cam, N_scan), order="F")

    return result


# ---------------------------------------------------------------------------
# Step 2 — Centre-crop camera frames
# ---------------------------------------------------------------------------

def crop_cam_images(cam_images: np.ndarray, N_d: int) -> np.ndarray:
    """Centre-crop each camera frame to N_d x N_d (mirrors MATLAB rounding)."""
    N = cam_images.shape[0]
    c = round(N / 2)
    half = round(N_d / 2)
    return cam_images[c - half: c + half, c - half: c + half, :]


# ---------------------------------------------------------------------------
# Optional Step 2.5 — Saturation repair (pattern-B masked median replacement)
# ---------------------------------------------------------------------------

def repair_saturation(
    cam_images: np.ndarray,
    abs_threshold: int = 25000,
    k_sigma: float = 6.0,
    box_h: int = 60,          # accepted but unused (kept for API compat)
    box_w: int = 80,          # accepted but unused (kept for API compat)
    kernel_size: int = 81,   # accepted but unused (kept for API compat)
    n_passes: int = 1,        # accepted but unused (kept for API compat)
    inplace: bool = True,
) -> tuple[np.ndarray, dict]:
    """Replace saturated (thresholded) pixels with the per-frame background mean.

    For each frame F = cam_images[:, :, i]:
        mask = (F > abs_threshold) | (F > F.mean() + k_sigma * F.std())
        if not mask.any(): skip
        bg = F[~mask].mean()    # background mean (unsaturated pixels only)
        F[mask] = bg            # replace only flagged pixels

    Clean frames (mask empty) are skipped — bit-identical to input.

    Notes
    -----
    - ``box_h``, ``box_w``, ``kernel_size``, ``n_passes`` are accepted
      for API compatibility but ignored. See git history for the box and
      median variants.

    Parameters
    ----------
    cam_images : ndarray [H, W, N_scan]  uint16
    abs_threshold : int
        Absolute pixel value above which a pixel is considered saturated.
    k_sigma : float
        Per-frame z-score factor for the second criterion (OR-combined).
    inplace : bool
        Modify cam_images directly (default True, to avoid 4+ GB copy).

    Returns
    -------
    cam_images : same array (modified) or a copy if ``inplace=False``
    stats : dict with n_frames_repaired and n_pixels_touched
    """
    if not inplace:
        cam_images = cam_images.copy()
    N = cam_images.shape[2]
    n_frames = 0
    n_px = 0
    for i in range(N):
        frame = cam_images[:, :, i]
        f64 = frame.astype(np.float64)
        mean = float(f64.mean())
        std = float(f64.std())
        mask = (frame > abs_threshold) | (frame > mean + k_sigma * std)
        if not mask.any():
            continue
        bg = float(f64[~mask].mean())
        frame[mask] = np.uint16(round(bg))
        n_frames += 1
        n_px += int(mask.sum())
    return cam_images, {"n_frames_repaired": n_frames, "n_pixels_touched": n_px}


# ---------------------------------------------------------------------------
# Step 3 — Pre-shifted 2-D FFT (batched)
# ---------------------------------------------------------------------------

def gen_fft_images(cam_images: np.ndarray) -> np.ndarray:
    """fftshift + fft2 along axes (0,1).  Matches MATLAB convention."""
    shifted = np.fft.fftshift(cam_images, axes=(0, 1))
    return np.fft.fft2(shifted, axes=(0, 1))


# ---------------------------------------------------------------------------
# Step 4 — Crop the off-axis sideband
# ---------------------------------------------------------------------------

def crop_fft_images(fft_images: np.ndarray, N_k: int, kx_c: int, ky_c: int) -> np.ndarray:
    """Extract a (2*N_k) x (2*N_k) region centred at (kx_c, ky_c) — 1-based MATLAB coords."""
    row_start = ky_c - N_k - 1
    row_end = ky_c + N_k - 1
    col_start = kx_c - N_k - 1
    col_end = kx_c + N_k - 1
    return fft_images[row_start:row_end, col_start:col_end, :]


# ---------------------------------------------------------------------------
# Step 5 — Off-axis demodulation
# ---------------------------------------------------------------------------

def gen_off_axis_images(cropped_fft_images: np.ndarray) -> np.ndarray:
    """ifftshift -> ifft2 -> fftshift -> horizontal flip  -> complex field."""
    result = np.fft.ifftshift(cropped_fft_images, axes=(0, 1))
    result = np.fft.ifft2(result, axes=(0, 1))
    result = np.fft.fftshift(result, axes=(0, 1))
    return result[:, ::-1, :]


# ---------------------------------------------------------------------------
# Steps 3 + 5 (combined, GPU)
# ---------------------------------------------------------------------------

def _gen_offaxis_gpu(
    cam_images: np.ndarray,
    N_k: int,
    kx_c: int,
    ky_c: int,
    batch_sz: int = 512,
) -> np.ndarray | None:
    """Steps 3 + 4 + 5 fused on GPU via torch.fft.

    Same math as ``gen_fft_images → crop_fft_images → gen_off_axis_images`` but
    fused: each chunk of frames goes uint16→float32→GPU→fft→crop, the sideband
    stays on GPU; once all chunks finish, ifft2+shift+flip runs in-chunks on
    GPU and the demodulated field comes back to CPU as complex64 once.

    Returns ``None`` if torch/CUDA isn't available — caller falls back to the
    original numpy path.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    device = torch.device('cuda')
    N_scan = cam_images.shape[2]
    row_start = ky_c - N_k - 1
    row_end = ky_c + N_k - 1
    col_start = kx_c - N_k - 1
    col_end = kx_c + N_k - 1

    # GPU sideband buffer in [B, H, W] layout for natural torch batching
    sb = torch.empty(
        (N_scan, 2 * N_k, 2 * N_k),
        dtype=torch.complex64,
        device=device,
    )

    # ---- Step 3 + 4: fft and sideband crop, in chunks ----
    for i in range(0, N_scan, batch_sz):
        end = min(i + batch_sz, N_scan)
        # uint16 -> float32 -> GPU. Transpose to [B, H, W] (contiguous copy).
        sl = np.ascontiguousarray(
            cam_images[:, :, i:end].astype(np.float32).transpose(2, 0, 1)
        )
        sl_t = torch.from_numpy(sl).to(device, non_blocking=True)
        sl_t = torch.fft.fftshift(sl_t, dim=(-2, -1))
        fft_t = torch.fft.fft2(sl_t, dim=(-2, -1))
        sb[i:end] = fft_t[:, row_start:row_end, col_start:col_end]
        del sl_t, fft_t

    # ---- Step 5: off-axis demod, in chunks (in-place into a fresh buffer) ----
    out_t = torch.empty_like(sb)
    for i in range(0, N_scan, batch_sz):
        end = min(i + batch_sz, N_scan)
        chunk = torch.fft.ifftshift(sb[i:end], dim=(-2, -1))
        chunk = torch.fft.ifft2(chunk, dim=(-2, -1))
        chunk = torch.fft.fftshift(chunk, dim=(-2, -1))
        out_t[i:end] = torch.flip(chunk, dims=(-1,))     # numpy's [:, ::-1, :]
        del chunk
    del sb

    # Back to [H, W, N_scan] for the rest of the pipeline (numpy).
    result = out_t.permute(1, 2, 0).contiguous().cpu().numpy()
    return result


# ---------------------------------------------------------------------------
# Step 6 — Assemble reflection matrix
# ---------------------------------------------------------------------------

def gen_rrmat(off_axis_images: np.ndarray, xc: int, yc: int) -> np.ndarray:
    """Assemble R(r_out; r_in) by placing each scan frame at its source offset.

    Optimised version: avoids per-frame np.pad/np.roll by writing each frame
    directly into the pre-allocated output array at the correct offset.

    Parameters
    ----------
    off_axis_images : [img_sz, img_sz, N_scan]  complex
    xc, yc          : 0-based PSF centre within the ROD

    Returns
    -------
    rinrout : [out_sz, out_sz, N_scan]  complex64
    """
    num_scan = off_axis_images.shape[2]
    sz = int(round(np.sqrt(num_scan)))
    rod_size = off_axis_images.shape[1]
    pad = max(abs(rod_size - 1 - xc), abs(rod_size - 1 - yc), xc, yc)
    sz_diff = sz - rod_size
    out_sz = rod_size + 2 * pad + sz_diff
    rinrout = np.zeros((out_sz, out_sz, num_scan), dtype=np.complex64)

    # Offset bookkeeping (even / odd sz_diff parity)
    correction = 2 * (sz_diff // 2) - sz_diff
    base_row = pad + correction - yc
    base_col = pad + correction - xc

    scan_rows = np.arange(num_scan) % sz
    scan_cols = np.arange(num_scan) // sz
    row_origins = base_row + scan_rows
    col_origins = base_col + scan_cols

    for i in range(num_scan):
        r0 = row_origins[i]
        c0 = col_origins[i]
        rinrout[r0:r0 + rod_size, c0:c0 + rod_size, i] = off_axis_images[:, :, i]

    return rinrout


# ---------------------------------------------------------------------------
# Step 7 — Optional ROI crop
# ---------------------------------------------------------------------------

def crop_roi(full_rr_mat: np.ndarray, xc: int, yc: int, Nx: int) -> np.ndarray:
    """Crop an Nx x Nx scan ROI centred at (xc, yc) on the scan grid.

    Parameters
    ----------
    full_rr_mat : [out_sz, out_sz, N_scan]  complex
    xc, yc      : 1-based ROI centre on the sz x sz scan grid
    Nx          : ROI side length (number of scan positions per axis)

    Returns
    -------
    [out_sz, out_sz, Nx**2]  complex
    """
    num_scan = full_rr_mat.shape[2]
    sz = int(round(np.sqrt(num_scan)))
    rod_size = full_rr_mat.shape[1]
    mat2d = full_rr_mat.reshape((rod_size ** 2, num_scan), order="F")

    half = round(Nx / 2)
    tmp = np.zeros((sz, sz), dtype=bool)
    row0 = yc - half
    col0 = xc - half
    tmp[row0: row0 + Nx, col0: col0 + Nx] = True
    roi = tmp.flatten(order="F")
    rr_mat = mat2d[:, roi]
    return rr_mat.reshape((rod_size, rod_size, Nx ** 2), order="F")


# ---------------------------------------------------------------------------
# Step 8 — Save rrmat
# ---------------------------------------------------------------------------

_MAT_V5_LIMIT = 2 * 1024 ** 3  # 2 GB — scipy MATLAB v5 hard limit


def save_rrmat(rrmat: np.ndarray, source_filepath: str, suffix: str = "rrmat_fullroi") -> str:
    """Save rrmat as .mat in a folder named after the source .mat (next to source).

    Layout:
        <folder>/<base_name>/<suffix>.mat
    where ``folder = dirname(source_filepath)`` and
    ``base_name = basename_without_ext(source_filepath)``.

    Uses MATLAB v5 (scipy) for arrays <= 2 GB, otherwise MATLAB v7.3 / HDF5
    via ``hdf5storage`` (loadable by MATLAB's standard ``load()``).
    """
    source_abs = os.path.abspath(source_filepath)
    folder = os.path.dirname(source_abs)
    base_name = os.path.splitext(os.path.basename(source_abs))[0]
    save_dir = os.path.join(folder, base_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{suffix}.mat")

    if os.path.abspath(save_path) == source_abs:
        raise ValueError(
            f"Computed save path would overwrite the source file:\n  {source_abs}"
        )

    nbytes = rrmat.nbytes
    if nbytes <= _MAT_V5_LIMIT:
        scipy.io.savemat(save_path, {"rrmat": rrmat})
        logger.info("rrmat saved to: %s  (%.1f MB, MATLAB v5)",
                    save_path, nbytes / 1024 ** 2)
    else:
        try:
            import hdf5storage
        except ImportError:
            raise ImportError(
                f"rrmat is {nbytes / 1024**3:.2f} GB (> 2 GB scipy limit).\n"
                "Install hdf5storage for MATLAB v7.3 support:  pip install hdf5storage"
            )
        hdf5storage.savemat(save_path, {"rrmat": rrmat})
        logger.info("rrmat saved to: %s  (%.2f GB, MATLAB v7.3)",
                    save_path, nbytes / 1024 ** 3)

    return save_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _get_int(struct, field):
    """Read a numeric field from a scipy mat_struct / SimpleNamespace as int."""
    val = getattr(struct, field)
    return int(val.flat[0]) if hasattr(val, "flat") else int(val)


def run_raw_to_rrmat(
    raw_mat_path: str | Path,
    N_d: int = 260,
    xc: int = 129,
    yc: int = 104,
    Nx_roi: int | None = None,
    save: bool = True,
    sat_repair: bool = False,
    sat_abs_threshold: int = 25000,
    sat_k_sigma: float = 6.0,
    sat_kernel_size: int = 81,
    sat_n_passes: int = 1,
    kx_c_override: int | None = None,
    ky_c_override: int | None = None,
) -> dict:
    """Execute the full raw-data -> rrmat pipeline.

    Parameters
    ----------
    raw_mat_path : str or Path
        Source .mat file containing scan metadata.  Companion .bin (named by
        ``RawDataFileName`` inside the .mat) must be alongside it.
    N_d : int, default 260
        ROD size in pixels — camera frames are centre-cropped to N_d x N_d.
        Matches the ESCLASS_GUI "ROD [px]" field.
    xc, yc : int, default 129, 104
        1-based scan centre on the camera frame (ESCLASS_GUI xc / yc fields).
    Nx_roi : int or None, default None
        If set, crop the rrmat to a square ROI of ``Nx_roi`` scan positions
        per axis (centred on the scan grid).  Otherwise the full rrmat is used.
    save : bool, default True
        If True, save the rrmat next to the source .mat (see ``save_rrmat``).

    Returns
    -------
    dict with keys:
        rrmat     : np.ndarray — final reflection matrix (shape [rod**2, Nx**2])
        save_path : str or None — path of the saved file (None if save=False)
        off_axis_params : struct with N_cam, N_K_max, kx_center, ky_center
    """
    raw_mat_path = str(raw_mat_path)
    logger.info("[1/8] Loading raw CASS data: %s", raw_mat_path)
    raw = load_cass_raw_data(raw_mat_path)
    cam_images = raw["cam_images"]
    off_axis_params = raw["off_axis_params"]

    N_k_raw = _get_int(off_axis_params, "N_K_max")
    kx_c_raw = _get_int(off_axis_params, "kx_center")
    ky_c_raw = _get_int(off_axis_params, "ky_center")

    # Scale carrier pixel positions from full-frame to cropped-frame coords.
    N_cam = cam_images.shape[0]
    scale = N_d / N_cam
    kx_c = int(round(kx_c_raw * scale)) - 1
    ky_c = int(round(ky_c_raw * scale))
    N_k = (round(N_k_raw / 2) + 2) * scale
    N_k = int(2 * np.floor(N_k / 2))

    if kx_c_override is not None:
        logger.info("kx_c override (scaled, on N_d frame): %d -> %d", kx_c, kx_c_override)
        kx_c = int(kx_c_override)
    if ky_c_override is not None:
        logger.info("ky_c override (scaled, on N_d frame): %d -> %d", ky_c, ky_c_override)
        ky_c = int(ky_c_override)

    if N_d is not None and N_d < cam_images.shape[1]:
        sz_diff = N_cam - N_d
        logger.info("[2/8] Centre-cropping camera images to %dx%d", N_d, N_d)
        cam_images = crop_cam_images(cam_images, N_d)
        xc = round(xc - sz_diff // 2)
        yc = round(yc - sz_diff // 2)
    else:
        logger.info("[2/8] Skipping spatial crop (N_d=%s)", N_d)

    logger.info("    cam_images %s  N_k=%d  kx_c=%d  ky_c=%d  (raw %d,%d)",
                cam_images.shape, N_k, kx_c, ky_c, kx_c_raw, ky_c_raw)

    # ---- Optional saturation repair (between spatial crop and FFT) ----
    if sat_repair:
        logger.info(
            "[2.5/8] Saturation repair (abs=%d, k_sigma=%.1f, kernel=%d, passes=%d) ...",
            sat_abs_threshold, sat_k_sigma, sat_kernel_size, sat_n_passes,
        )
        cam_images, sat_stats = repair_saturation(
            cam_images,
            abs_threshold=sat_abs_threshold,
            k_sigma=sat_k_sigma,
            kernel_size=sat_kernel_size,
            n_passes=sat_n_passes,
            inplace=True,
        )
        logger.info(
            "    repaired %d/%d frames, %d pixels touched",
            sat_stats["n_frames_repaired"], cam_images.shape[2],
            sat_stats["n_pixels_touched"],
        )
    else:
        logger.info("[2.5/8] Saturation repair disabled (sat_repair=False)")

    # ---- Steps 3 + 4 + 5 (combined; GPU when available, numpy fallback) ----
    N_scan = cam_images.shape[2]
    off_axis = _gen_offaxis_gpu(cam_images, N_k, kx_c, ky_c)
    if off_axis is not None:
        logger.info("[3-4/8] FFT + sideband crop + off-axis demod (GPU, fused)")
        logger.info("    off_axis shape %s", off_axis.shape)
    else:
        logger.info("[3/8] FFT + sideband crop (CPU, batched) ...")
        fft_imgs = np.empty([2 * N_k, 2 * N_k, N_scan], dtype=np.complex64)
        batch_sz = 64
        for i in range(0, N_scan, batch_sz):
            sl = cam_images[:, :, i: i + batch_sz].astype(np.float32)
            tmp = gen_fft_images(sl).astype(np.complex64)
            fft_imgs[:, :, i: i + batch_sz] = crop_fft_images(tmp, N_k, kx_c, ky_c)
        logger.info("    fft shape %s", fft_imgs.shape)
        logger.info("[4/8] Off-axis demodulation -> complex field")
        off_axis = gen_off_axis_images(fft_imgs).astype(np.complex64)
        del fft_imgs

    # ---- Step 6 ----
    logger.info("[5/8] Assembling rrmat from off_axis %s", off_axis.shape)
    sz_off_axis = off_axis.shape[1]
    xc_off_axis = round((N_cam - xc) * sz_off_axis / N_cam)
    yc_off_axis = round(yc * sz_off_axis / N_cam)
    rinrout = gen_rrmat(off_axis, xc_off_axis, yc_off_axis)
    logger.info("    rrmat shape %s  dtype %s", rinrout.shape, rinrout.dtype)

    # ---- Step 7 (optional ROI crop) ----
    if Nx_roi is not None:
        N_x = int(np.sqrt(rinrout.shape[2]))
        xc_roi = round(N_x / 2)
        yc_roi = round(N_x / 2)
        logger.info("[6/8] ROI crop: centre=(%d,%d)  Nx=%d", xc_roi, yc_roi, Nx_roi)
        rrmat = crop_roi(rinrout, xc_roi, yc_roi, Nx_roi)
    else:
        logger.info("[6/8] No ROI crop — using full rrmat")
        rrmat = rinrout

    # Reshape to (out_sz**2, N_x**2) to match the 2-D MATLAB layout used by solver.
    N_x = int(np.sqrt(rrmat.shape[2]))
    rrmat = rrmat.reshape((-1, N_x ** 2), order="F")
    logger.info("[7/8] Final rrmat shape %s", rrmat.shape)

    # ---- Step 8 (save) ----
    save_path = None
    if save:
        logger.info("[8/8] Saving rrmat ...")
        save_path = save_rrmat(rrmat, raw_mat_path, suffix="rrmat_fullroi")
    else:
        logger.info("[8/8] Save skipped (save=False)")

    return {
        "rrmat": rrmat,
        "save_path": save_path,
        "off_axis_params": off_axis_params,
    }


# ---------------------------------------------------------------------------
# Convenience helper for solver integration
# ---------------------------------------------------------------------------

def ensure_rrmat_exists(
    rrmat_path: str | Path,
    raw_mat_path: str | Path | None = None,
    raw_N_d: int = 260,
    raw_xc: int = 129,
    raw_yc: int = 104,
    raw_Nx_roi: int | None = None,
    sat_repair: bool = False,
    sat_abs_threshold: int = 25000,
    sat_k_sigma: float = 6.0,
    sat_kernel_size: int = 81,
    sat_n_passes: int = 1,
) -> str:
    """Return a path to a valid rrmat .mat.  Generate from raw data if missing.

    - If ``rrmat_path`` exists: return it unchanged.
    - Else if ``raw_mat_path`` is given and exists: run the raw-to-rrmat
      pipeline, which saves to  ``<raw_dir>/<raw_basename>/rrmat_fullroi.mat``,
      and return that generated path.
    - Else: raise FileNotFoundError with a clear message.

    The caller should treat the returned value as the authoritative rrmat path
    from that point on (it may differ from the original ``rrmat_path`` if
    generation was triggered).
    """
    if rrmat_path and Path(rrmat_path).exists():
        return str(rrmat_path)

    if raw_mat_path and Path(raw_mat_path).exists():
        logger.info(
            "rrmat not found at %s — generating from raw data at %s",
            rrmat_path, raw_mat_path,
        )
        result = run_raw_to_rrmat(
            raw_mat_path,
            N_d=raw_N_d,
            xc=raw_xc,
            yc=raw_yc,
            Nx_roi=raw_Nx_roi,
            save=True,
            sat_repair=sat_repair,
            sat_abs_threshold=sat_abs_threshold,
            sat_k_sigma=sat_k_sigma,
            sat_kernel_size=sat_kernel_size,
            sat_n_passes=sat_n_passes,
        )
        return result["save_path"]

    raise FileNotFoundError(
        f"rrmat not found at '{rrmat_path}' and no valid raw_mat_path provided "
        f"(raw_mat_path='{raw_mat_path}'). Set either rrmat_path to an existing "
        f".mat file or raw_mat_path to a CASS raw .mat to auto-generate the rrmat."
    )
