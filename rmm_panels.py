"""Stage-4b — manuscript panels rendered from the pipeline's own artifacts.

Self-contained, pruned port of the workspace-root ``render_paper_panels.py``
(which depends on ``utils/``). This copy carries only the six panels the
rmm2odt pipeline can produce end-to-end, and drops every ``utils``-backed
path (the panel_3d Rxk/rrmat comparison rows in particular). Matplotlib +
numpy only, plus scipy/h5py to read ``.mat`` files.

Panels (each maps 1:1 to a file in manuscript_assets/svg_panels_sample5_251208):
  fig2c_confocal_reflectance         confocal image of the raw rrmat measurement
  fig2e_optimized_layers             per-layer reconstruction grid (amp row + phase row)
  fig2e_optimized_object             reconstructed sample plane (amp | phase)
  fig3b_depth_scanned_layers         per-depth amp/phase montage of the depth scan
  fig3d_angular_transmission_fields  odt_input.mat content (illum phase / output amp+phase)
  fig4f_multi_depth_xy_odt_images    K XY cross-sections of the ODT volume (gray, z=15/20/25 µm)

The single public entry point is :func:`render_manuscript_panels`; each panel
is guarded so a missing input (or a failing panel) is warned-and-skipped
rather than aborting the rest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

LOG = logging.getLogger("rmm2odt.panels")

# Colour-map convention (consistent across panels), matching render_paper_panels.
CMAP_PHASE = "jet"

PUBLICATION_STYLE = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        9,
    "axes.linewidth":   0.8,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "svg.fonttype":     "none",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "image.interpolation": "nearest",
    "image.cmap":       "viridis",
}


def _apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_STYLE)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _glob_layer_files(layer_dir: Path, epoch: int) -> list[Path]:
    pattern = f"output_layer_*_epoch_{epoch}.txt"
    return sorted(layer_dir.glob(pattern), key=lambda p: int(p.stem.split("_")[2]))


def _load_complex_layers(layer_dir: Path, epoch: int) -> list[np.ndarray]:
    files = _glob_layer_files(layer_dir, epoch)
    if not files:
        raise FileNotFoundError(f"no output_layer_*_epoch_{epoch}.txt in {layer_dir}")
    layers = [np.loadtxt(str(f), dtype=np.complex64) for f in files]
    LOG.info("loaded %d layer file(s) from %s (epoch=%d)", len(layers), layer_dir, epoch)
    return layers


def _load_mat(path: Path, key: str | None = None):
    import scipy.io as sio
    try:
        d = sio.loadmat(str(path))
    except (NotImplementedError, ValueError):
        import h5py
        d = {}
        with h5py.File(str(path), "r") as f:
            for k in f.keys():
                arr = f[k][...]
                if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
                    arr = arr["real"] + 1j * arr["imag"]
                if arr.ndim >= 2:
                    arr = np.transpose(arr, list(range(arr.ndim - 1, -1, -1)))
                d[k] = arr
    if key is not None:
        if key not in d:
            avail = ", ".join(k for k in d if not k.startswith("__"))
            raise KeyError(f"key {key!r} not in {path.name}; available: {avail}")
        return d[key]
    return d


def _load_ss_opt_volume(ss_opt_mat: Path, ss_opt_key: str) -> np.ndarray:
    """Load a 3-D RI volume and apply the rODT_vis display transform:
    lateral fftshift, transpose to Z-first, flip Z. (Ny,Nx,Nz) -> (Nz,Ny,Nx)."""
    if not Path(ss_opt_mat).exists():
        raise FileNotFoundError(f"{ss_opt_mat} not found")
    vol = _load_mat(Path(ss_opt_mat), ss_opt_key)
    if np.iscomplexobj(vol):
        vol = vol.real
    if vol.ndim != 3:
        raise ValueError(f"expected 3-D ss_opt, got {vol.shape}")
    vol = np.fft.fftshift(np.fft.fftshift(vol, axes=1), axes=0)
    vol = np.transpose(vol, (2, 0, 1))
    vol = np.flip(vol, axis=0)
    return np.ascontiguousarray(vol)


# ---------------------------------------------------------------------------
# Style / drawing helpers
# ---------------------------------------------------------------------------

def _strip_ticks(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _add_scalebar(ax, pixel_size_um: float, length_um: float,
                  *, location: str = "lower right", color: str = "white",
                  show_label: bool = False):
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(
            pixel_size_um, "um", location=location, frameon=False, color=color,
            fixed_value=length_um, scale_loc=("top" if show_label else "none"),
            width_fraction=0.04, font_properties={"size": 8, "weight": "bold"}))
    except ImportError:
        xlim = ax.get_xlim()
        img_w_px = abs(xlim[1] - xlim[0])
        bar_frac = min((length_um / pixel_size_um) / img_w_px, 0.4)
        x1 = 0.95; x0 = x1 - bar_frac; y = 0.07
        ax.plot([x0, x1], [y, y], color=color, lw=2.5, transform=ax.transAxes,
                clip_on=True, solid_capstyle="butt")
        if show_label:
            ax.text((x0 + x1) / 2, y + 0.025, f"{length_um:.0f} μm", color=color,
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    transform=ax.transAxes)


def _attach_colorbar(fig, ax, im, *, size: str = "4%", pad: float = 0.05,
                     phase_ticks: bool = False):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    if phase_ticks:
        cb.set_ticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
        cb.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    return cb


def _save(fig, out_dir: Path, stem: str, formats: Iterable[str]):
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = out_dir / f"{stem}.{fmt}"
        fig.savefig(out, format=fmt)
        written.append(str(out))
        LOG.info("wrote %s", out)
    plt.close(fig)
    return written


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    r0 = h // 2 - size // 2
    c0 = w // 2 - size // 2
    return arr[..., r0:r0 + size, c0:c0 + size]


def _circ_mask_2d(img: np.ndarray, fill_value=np.nan) -> np.ndarray:
    out = np.array(img, copy=True)
    h, w = out.shape[-2:]
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = min(cy, cx)
    outside = (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
    out[..., outside] = fill_value
    return out


def _resolve_outline_palette(colors, n):
    if colors is None:
        colors = (["#e63946", "#3a86ff"] if n == 2 else
                  ["#e63946", "#2a9d8f", "#3a86ff", "#ff9800", "#9c27b0"][:max(n, 1)])
    if len(colors) < n:
        colors = list(colors) + [colors[-1]] * (n - len(colors))
    return list(colors)


def _parse_layer_positions_from_readme(layer_dir: Path):
    import re
    readme = Path(layer_dir) / "readme.txt"
    if not readme.exists():
        return None
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"^layer_positions:\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL)
    if m is None:
        m = re.search(r"layer_positions:\s*\[(.*?)\]", text, re.DOTALL)
    if m is None:
        return None
    inside = m.group(1).replace("\n", " ").strip()
    try:
        vals = [float(t) for t in inside.split() if t]
    except ValueError:
        return None
    return vals or None


def _img_arr2confocal(img_arr: np.ndarray) -> np.ndarray:
    """Confocal image = diag(R) reshaped onto the scan grid, from an
    (N_scan, H, W) reflection-matrix image stack (F-order per pixel)."""
    n_scan, h, w = img_arr.shape
    scan_size = int(np.sqrt(n_scan))
    if scan_size != h:
        img_arr = _center_crop(img_arr, scan_size)
        h = w = scan_size
    rr_matrix = np.zeros((h * w, n_scan), dtype=np.complex64)
    for i in range(n_scan):
        rr_matrix[:, i] = img_arr[i].reshape(h * w, order="F")
    return np.diag(rr_matrix).reshape((scan_size, scan_size), order="F")


def _demodulate_tilt(field: np.ndarray, kx_pp: float, ky_pp: float) -> np.ndarray:
    H, W = field.shape[-2:]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx -= W // 2
    yy -= H // 2
    phasor = np.exp(-1j * (kx_pp * xx + ky_pp * yy)).astype(np.complex64)
    return field * phasor


def _auto_demodulate_tilt(field: np.ndarray):
    """Estimate the field's dominant tilt from the FFT-magnitude peak
    (sub-pixel refined) and remove it. Returns (demod_field, kx_pp, ky_pp)."""
    H, W = field.shape[-2:]
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    mag = np.abs(F)
    iy, ix = np.unravel_index(int(np.argmax(mag)), F.shape)

    def _parab_off(a, b, c):
        denom = (a - 2.0 * b + c)
        if abs(denom) < 1e-30:
            return 0.0
        return float(max(-1.0, min(1.0, 0.5 * (a - c) / denom)))

    x_off = _parab_off(mag[iy, ix - 1], mag[iy, ix], mag[iy, ix + 1]) if 0 < ix < W - 1 else 0.0
    y_off = _parab_off(mag[iy - 1, ix], mag[iy, ix], mag[iy + 1, ix]) if 0 < iy < H - 1 else 0.0
    kx_pp = (ix + x_off - W // 2) * 2.0 * np.pi / W
    ky_pp = (iy + y_off - H // 2) * 2.0 * np.pi / H
    out = _demodulate_tilt(field, kx_pp, ky_pp)
    for _ in range(3):
        F2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(out)))
        mag2 = np.abs(F2)
        iy2, ix2 = np.unravel_index(int(np.argmax(mag2)), F2.shape)
        if abs(iy2 - H // 2) > 1 or abs(ix2 - W // 2) > 1:
            break
        x_off2 = _parab_off(mag2[iy2, ix2 - 1], mag2[iy2, ix2], mag2[iy2, ix2 + 1]) if 0 < ix2 < W - 1 else 0.0
        y_off2 = _parab_off(mag2[iy2 - 1, ix2], mag2[iy2, ix2], mag2[iy2 + 1, ix2]) if 0 < iy2 < H - 1 else 0.0
        d_kx = (ix2 + x_off2 - W // 2) * 2.0 * np.pi / W
        d_ky = (iy2 + y_off2 - H // 2) * 2.0 * np.pi / H
        if abs(d_kx) < 1e-5 and abs(d_ky) < 1e-5:
            break
        out = _demodulate_tilt(out, d_kx, d_ky)
        kx_pp += d_kx
        ky_pp += d_ky
    return out, float(kx_pp), float(ky_pp)


def _build_rxk_slices(rrmat_path, kxky_per_pixel, *, wavelength_um, medium_n,
                      resolution_um, propagate_distance_um=None, NA=None):
    """Angular images from a reflection matrix in the Rxk basis.

    Load the RM (``.mat`` via multilayer_torch.data_loading.load_rrmat, or a
    ``.npy`` rrmat / label stack), symmetrise to square, optionally propagate
    both planes by ``propagate_distance_um`` (ASM), transform the input axis
    to k-space (rm_to_xk), then for each per-pixel illumination increment
    ``(kx_pp, ky_pp)`` pick the matching k-column and reshape to a real-space
    output image. Returns ``(fields (n_angles, N, N) complex64, N)``."""
    from rmm_rm import symmetrize_rm, rm_to_xk, propage_rrmat, img_arr_to_rm
    rrmat_path = Path(rrmat_path)
    if rrmat_path.suffix.lower() == ".npy":
        raw = np.load(str(rrmat_path))
        if raw.ndim == 3:
            rr = img_arr_to_rm(raw, order="F")
        elif raw.ndim == 2:
            rr = raw
        else:
            raise ValueError(f"unsupported .npy ndim {raw.ndim}")
    else:
        from multilayer_torch.data_loading import load_rrmat
        rr, Nx_det, Nx_illum, *_ = load_rrmat(rrmat_path, wl=wavelength_um,
                                              n0=medium_n, dx=resolution_um)
        LOG.info("panel_3d/Rxk: rrmat %s Nx_det=%d Nx_illum=%d",
                 tuple(rr.shape), Nx_det, Nx_illum)
    rr_sq, info = symmetrize_rm(rr)
    LOG.info("panel_3d/Rxk: %s", info.summary())
    N = info.n_kept
    if propagate_distance_um:
        rr_sq = propage_rrmat(rr_sq, lmb=wavelength_um, n=medium_n,
                              d=propagate_distance_um, res=resolution_um, NA=NA)
        LOG.info("panel_3d/Rxk: propagated d=%+.3f um (lmb=%.3f n=%.2f res=%.4f NA=%s)",
                 propagate_distance_um, wavelength_um, medium_n, resolution_um,
                 "natural" if NA is None else f"{NA:.2f}")
    rxk = rm_to_xk(rr_sq, Nx_in=N, Ny_in=N)
    fields = np.empty((len(kxky_per_pixel), N, N), dtype=np.complex64)
    for i, (kx_pp, ky_pp) in enumerate(kxky_per_pixel):
        kx_idx = N // 2 + int(round(kx_pp * N / (2.0 * np.pi)))
        ky_idx = N // 2 + int(round(ky_pp * N / (2.0 * np.pi)))
        kx_idx = min(max(kx_idx, 0), N - 1)
        ky_idx = min(max(ky_idx, 0), N - 1)
        flat = ky_idx * N + kx_idx        # row-major flat (matches rm_to_xk cols)
        fields[i] = rxk[:, flat].reshape(N, N, order="F")
    return fields, N


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_1b(*, rrmat: Path, object_size: int = 120, resolution_um: float = 0.2581,
             scalebar_length: float = 10.0, wavelength_um: float = 0.517,
             medium_n: float = 1.33, out_dir: Path, formats: Iterable[str]):
    """Fig 1b — confocal image of the raw rrmat measurement (|E| + phase)."""
    import matplotlib.pyplot as plt
    rrmat = Path(rrmat)
    if not rrmat.exists():
        LOG.warning("panel_1b: %s not found; skipping.", rrmat)
        return
    if rrmat.suffix.lower() == ".npy":
        arr = np.load(str(rrmat))
    elif rrmat.suffix.lower() == ".mat":
        from multilayer_torch.data_loading import load_rrmat, rrmat_to_illum_labels
        rr, Nx_det, Nx_illum, *_ = load_rrmat(rrmat, wl=wavelength_um, n0=medium_n,
                                              dx=resolution_um)
        LOG.info("panel_1b: rrmat %s Nx_det=%d Nx_illum=%d", tuple(rr.shape), Nx_det, Nx_illum)
        _, arr = rrmat_to_illum_labels(rr, Nx_det, Nx_illum)
    else:
        raise ValueError(f"panel_1b: unsupported extension {rrmat.suffix!r}")
    if arr.ndim != 3:
        raise ValueError(f"panel_1b: expected (N_scan, H, W), got {arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)

    cfc = _img_arr2confocal(arr)
    cfc = _center_crop(cfc, object_size)
    amp = np.abs(cfc); amp = amp / (amp.max() + 1e-12)
    phs = np.angle(cfc)

    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                     gridspec_kw=dict(wspace=0.18))
    im_a = ax_a.imshow(amp, cmap="afmhot", vmin=0, vmax=1)
    ax_a.set_box_aspect(1.0); _strip_ticks(ax_a); ax_a.set_title("|E|", pad=4)
    _add_scalebar(ax_a, resolution_um, scalebar_length)
    _attach_colorbar(fig, ax_a, im_a)
    im_p = ax_p.imshow(phs, cmap=CMAP_PHASE, vmin=-np.pi, vmax=np.pi)
    ax_p.set_box_aspect(1.0); _strip_ticks(ax_p); ax_p.set_title("phase", pad=4)
    _attach_colorbar(fig, ax_p, im_p, phase_ticks=True)
    return _save(fig, out_dir, "fig2c_confocal_reflectance", formats)


def panel_2c(*, layer_dir: Path, epoch: int, depths_um=None, resolution_um: float = 0.2581,
             scalebar_length: float = 10.0, crop_size=None, apply_circular_mask: bool = True,
             exclude_last: bool = True, out_dir: Path, formats: Iterable[str]):
    """Fig 2c — per-layer reconstruction grid (amp row = gray, phase row = jet)."""
    import matplotlib.pyplot as plt
    layers = _load_complex_layers(Path(layer_dir), epoch)
    if exclude_last:
        layers = layers[:-1] if len(layers) > 1 else layers
    if not layers:
        LOG.warning("panel_2c: no layers after excluding sample; skipping")
        return
    if crop_size is None:
        layers_cropped = [_center_crop(L, L.shape[-1] // 2) for L in layers]
    else:
        layers_cropped = [_center_crop(L, crop_size) for L in layers]
    n = len(layers_cropped)

    if depths_um is None:
        depths_um = _parse_layer_positions_from_readme(Path(layer_dir))
        if depths_um is not None and exclude_last and len(depths_um) >= n + 1:
            depths_um = depths_um[:-1]
    titles = ([f"z = {z:.1f} μm" for z in depths_um[:n]] if depths_um is not None
              else [f"layer {i+1}" for i in range(n)])

    amps = [np.abs(L) / (np.abs(L).max() + 1e-12) for L in layers_cropped]
    phases = [np.angle(L) for L in layers_cropped]
    if apply_circular_mask:
        amps = [_circ_mask_2d(a, fill_value=0.1) for a in amps]
        amps = [a / (np.nanmax(a) + 1e-12) for a in amps]
        phases = [_circ_mask_2d(p, fill_value=0.0) for p in phases]

    fig_w = 6.85
    panel_w = fig_w / n
    fig = plt.figure(figsize=(fig_w, panel_w * 2 * 1.05))
    gs = fig.add_gridspec(2, n + 1, hspace=0.04, wspace=0.04,
                          width_ratios=[1.0] * n + [0.05],
                          left=0.01, right=0.99, top=0.95, bottom=0.02)
    last_im_a = last_im_p = None
    for i, (amp, phs) in enumerate(zip(amps, phases)):
        ax_a = fig.add_subplot(gs[0, i])
        last_im_a = ax_a.imshow(amp, cmap="gray", vmin=0, vmax=1)
        ax_a.set_box_aspect(1.0); _strip_ticks(ax_a); ax_a.set_title(titles[i], pad=2)
        _add_scalebar(ax_a, resolution_um, scalebar_length)
        ax_p = fig.add_subplot(gs[1, i])
        last_im_p = ax_p.imshow(phs, cmap=CMAP_PHASE, vmin=-np.pi, vmax=np.pi)
        ax_p.set_box_aspect(1.0); _strip_ticks(ax_p)
    if last_im_a is not None:
        cb_a = fig.colorbar(last_im_a, cax=fig.add_subplot(gs[0, n]))
        cb_a.ax.tick_params(labelsize=7)
        cb_p = fig.colorbar(last_im_p, cax=fig.add_subplot(gs[1, n]))
        cb_p.ax.tick_params(labelsize=7)
        cb_p.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb_p.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    return _save(fig, out_dir, "fig2e_optimized_layers", formats)


def panel_2d(*, layer_dir: Path, epoch: int, resolution_um: float = 0.2581,
             scalebar_length: float = 10.0, crop_size=None, out_dir: Path,
             formats: Iterable[str]):
    """Fig 2d — reconstructed sample plane: amplitude | phase."""
    import matplotlib.pyplot as plt
    layers = _load_complex_layers(Path(layer_dir), epoch)
    L = layers[-1]
    if crop_size is not None:
        L = _center_crop(L, crop_size)
    amp = np.abs(L); amp = amp / (amp.max() + 1e-12)
    phs = np.angle(L)
    fig, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                     gridspec_kw=dict(wspace=0.18))
    im_a = ax_a.imshow(amp, cmap="afmhot", vmin=0, vmax=1)
    _strip_ticks(ax_a); ax_a.set_title("|E|", pad=4)
    _add_scalebar(ax_a, resolution_um, scalebar_length)
    _attach_colorbar(fig, ax_a, im_a)
    im_p = ax_p.imshow(phs, cmap=CMAP_PHASE, vmin=-np.pi, vmax=np.pi)
    _strip_ticks(ax_p); ax_p.set_title("phase", pad=4)
    _attach_colorbar(fig, ax_p, im_p, phase_ticks=True)
    return _save(fig, out_dir, "fig2e_optimized_object", formats)


def panel_3b(*, scan_stack: Path, depth_indices=None, depth_axis_um=None,
             resolution_um: float = 0.2581, scalebar_length: float = 10.0,
             crop_size=None, out_dir: Path, formats: Iterable[str]):
    """Fig 3b — depth-scan montage: amp row (inferno) + phase row (jet), K depths."""
    import matplotlib.pyplot as plt
    scan_path = Path(scan_stack)
    if not scan_path.exists():
        LOG.warning("panel_3b: %s not found; skipping.", scan_stack)
        return
    arr = np.load(str(scan_path))
    if arr.ndim != 3:
        raise ValueError(f"panel_3b: expected (N,H,W), got {arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)
    n_depths = arr.shape[0]

    if depth_axis_um is None:
        sib = scan_path.parent / "depths.npy"
        if sib.exists():
            depth_axis_um = list(np.load(str(sib)).astype(float))
            LOG.info("panel_3b: depth axis from %s (%.1f..%.1f μm)",
                     sib, depth_axis_um[0], depth_axis_um[-1])
    if depth_indices is None:
        k = min(6, n_depths)
        depth_indices = list(np.linspace(0, n_depths - 1, k).astype(int))
    depth_indices = [max(0, min(n_depths - 1, int(i))) for i in depth_indices]
    if depth_axis_um is not None:
        labels = [f"z = {depth_axis_um[i]:+.1f} μm" for i in depth_indices]
    else:
        labels = [f"idx {i}" for i in depth_indices]

    if crop_size is not None:
        arr = np.stack([_center_crop(arr[i], crop_size) for i in range(n_depths)])
    n = len(depth_indices)
    amp_max = np.abs(arr).max()

    fig_w = 6.85
    panel_w = fig_w / n
    fig = plt.figure(figsize=(fig_w, panel_w * 2 * 1.05))
    gs = fig.add_gridspec(2, n + 1, hspace=0.04, wspace=0.04,
                          width_ratios=[1.0] * n + [0.05],
                          left=0.01, right=0.99, top=0.95, bottom=0.02)
    last_im_a = last_im_p = None
    for col, di in enumerate(depth_indices):
        field = arr[di]
        amp = np.abs(field) / (amp_max + 1e-12)
        phs = np.angle(field)
        ax_a = fig.add_subplot(gs[0, col])
        last_im_a = ax_a.imshow(amp, cmap="inferno", vmin=0, vmax=1)
        _strip_ticks(ax_a); ax_a.set_title(labels[col], pad=2)
        if col == 0:
            _add_scalebar(ax_a, resolution_um, scalebar_length)
        ax_p = fig.add_subplot(gs[1, col])
        last_im_p = ax_p.imshow(phs, cmap=CMAP_PHASE, vmin=-np.pi, vmax=np.pi)
        _strip_ticks(ax_p)
    if last_im_a is not None:
        cb_a = fig.colorbar(last_im_a, cax=fig.add_subplot(gs[0, n]))
        cb_a.ax.tick_params(labelsize=7)
        cb_p = fig.colorbar(last_im_p, cax=fig.add_subplot(gs[1, n]))
        cb_p.ax.tick_params(labelsize=7)
        cb_p.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb_p.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    return _save(fig, out_dir, "fig3b_depth_scanned_layers", formats)


def panel_3d(*, odt_input_mat: Path, angle_indices=None, resolution_um: float = 0.2581,
             scalebar_length: float = 10.0, crop_size=None, wavelength_um: float = 0.517,
             medium_n: float = 1.33, demodulate_tilt: bool = True, show_output_amp: bool = True,
             subtract_output_ramp: bool = False,
             rrmat_path=None, rrmat_propagate_d=None, rrmat_NA=None,
             out_dir: Path, formats: Iterable[str]):
    """Fig 3d — odt_input.mat content, optionally with propagated-rrmat rows.

    Without ``rrmat_path``: 3 rows × K angles — illumination phase / GVAS
    output |E| / GVAS output phase.

    With ``rrmat_path``: 5 rows — adds the angular image extracted from the
    reflection matrix (propagated by ``rrmat_propagate_d`` µm, ASM) at the
    same illumination k-vector: rows 3-4 = Rxk |E| / Rxk phase. GVAS phase is
    demodulated by the illumination tilt; Rxk phase gets a second residual-tilt
    demod stage."""
    import matplotlib.pyplot as plt
    if not Path(odt_input_mat).exists():
        LOG.warning("panel_3d: %s not found; skipping.", odt_input_mat)
        return
    d = _load_mat(Path(odt_input_mat))
    illum_key = "Illumination_stack" if "Illumination_stack" in d else "UI_stack_In"
    output_key = "Output_stack" if "Output_stack" in d else "U_stack_In"
    UI = np.asarray(d[illum_key]); U = np.asarray(d[output_key])

    def _to_AHW(x):
        if x.ndim != 3:
            raise ValueError(f"expected 3-D stack, got {x.shape}")
        if x.shape[1] == x.shape[2] and x.shape[0] != x.shape[1]:
            pass
        elif x.shape[0] == x.shape[1] and x.shape[2] != x.shape[0]:
            x = np.transpose(x, (2, 0, 1))
        return x if np.iscomplexobj(x) else x.astype(np.complex64)
    UI = _to_AHW(UI); U = _to_AHW(U)
    n_angles = UI.shape[0]

    if angle_indices is None:
        k = min(5, n_angles)
        angle_indices = list(np.linspace(0, n_angles - 1, k).astype(int))
    angle_indices = [max(0, min(n_angles - 1, int(a))) for a in angle_indices]

    H_full, W_full = UI.shape[-2:]
    cy_f, cx_f = H_full // 2, W_full // 2
    NA_FACTOR = wavelength_um / (2.0 * np.pi * resolution_um)
    angle_labels = []
    kxky_per_pixel = []
    for i, ai in enumerate(angle_indices):
        kx_pp = float(np.angle(UI[ai, cy_f, cx_f + 1] * UI[ai, cy_f, cx_f].conj()))
        ky_pp = float(np.angle(UI[ai, cy_f + 1, cx_f] * UI[ai, cy_f, cx_f].conj()))
        kxky_per_pixel.append((kx_pp, ky_pp))
        NA_x, NA_y = kx_pp * NA_FACTOR, ky_pp * NA_FACTOR
        angle_labels.append((f"$(k_x, k_y) = ({NA_x:+.2f}, {NA_y:+.2f})$ NA" if i == 0
                             else f"$({NA_x:+.2f}, {NA_y:+.2f})$ NA"))

    # Optional: angular images from the propagated reflection matrix.
    rxk_fields = None
    if rrmat_path is not None and Path(rrmat_path).exists():
        try:
            rxk_fields, _N = _build_rxk_slices(
                rrmat_path, kxky_per_pixel, wavelength_um=wavelength_um,
                medium_n=medium_n, resolution_um=resolution_um,
                propagate_distance_um=rrmat_propagate_d, NA=rrmat_NA)
        except Exception as e:
            LOG.error("panel_3d: Rxk rows failed (%s); rendering 3-row fig3d", e)
            rxk_fields = None
    elif rrmat_path is not None:
        LOG.warning("panel_3d: rrmat %s not found; skipping Rxk rows.", rrmat_path)

    if crop_size is not None:
        UI = np.stack([_center_crop(UI[i], crop_size) for i in range(n_angles)])
        U = np.stack([_center_crop(U[i], crop_size) for i in range(n_angles)])
        if rxk_fields is not None:
            rxk_fields = np.stack([_center_crop(rxk_fields[i], crop_size)
                                   for i in range(rxk_fields.shape[0])])

    n = len(angle_indices)
    # Build the active row list (top -> bottom). Each: (key, label, cmap, vmin, vmax, phase_ticks).
    out_tag = "GVAS" if rxk_fields is not None else "Output"
    rows = [("illum_ph", "Illum. phase", CMAP_PHASE, -np.pi, np.pi, True)]
    if show_output_amp:
        rows.append(("out_amp", f"{out_tag} |E|", "hot", 0.0, 1.0, False))
    rows.append(("out_ph", f"{out_tag} phase", CMAP_PHASE, -np.pi, np.pi, True))
    if rxk_fields is not None:
        d_lbl = f" (+{rrmat_propagate_d:g} µm)" if rrmat_propagate_d else ""
        rows.append(("rxk_amp", f"Rxk |E|{d_lbl}", "hot", 0.0, 1.0, False))
        rows.append(("rxk_ph", "Rxk phase", CMAP_PHASE, -np.pi, np.pi, True))
    n_rows = len(rows)

    fig_w = 6.85
    LEFT, RIGHT, TOP, BOTTOM = 0.07, 0.99, 0.93, 0.03
    col_w_frac = (RIGHT - LEFT) / (n + 0.05)
    row_h_frac = (TOP - BOTTOM) / n_rows
    fig = plt.figure(figsize=(fig_w, fig_w * col_w_frac / row_h_frac))
    gs = fig.add_gridspec(n_rows, n + 1, hspace=0.10, wspace=0.04,
                          width_ratios=[1.0] * n + [0.05],
                          left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)

    last_im: dict = {}
    for col, ai in enumerate(angle_indices):
        if demodulate_tilt:
            _, kx_i, ky_i = _auto_demodulate_tilt(UI[ai])
        else:
            kx_i = ky_i = 0.0
        imgs = {"illum_ph": np.angle(UI[ai])}
        if show_output_amp:
            a = np.abs(U[ai]); imgs["out_amp"] = a / (a.max() + 1e-12)
        U_out = U[ai]
        if demodulate_tilt:
            U_out = _demodulate_tilt(U_out, kx_i, ky_i)     # cancel illumination tilt
        if subtract_output_ramp:
            U_out, _, _ = _auto_demodulate_tilt(U_out)      # subtract residual linear phase ramp
        imgs["out_ph"] = np.angle(U_out)
        if rxk_fields is not None:
            xk = rxk_fields[col]
            ax_ = np.abs(xk); imgs["rxk_amp"] = ax_ / (ax_.max() + 1e-12)
            if demodulate_tilt:
                xk2, _, _ = _auto_demodulate_tilt(_demodulate_tilt(xk, kx_i, ky_i))
                imgs["rxk_ph"] = np.angle(xk2)
            else:
                imgs["rxk_ph"] = np.angle(xk)
        for r, (key, lbl, cmap, vmn, vmx, pt) in enumerate(rows):
            ax = fig.add_subplot(gs[r, col])
            im = ax.imshow(imgs[key], cmap=cmap, vmin=vmn, vmax=vmx)
            ax.set_box_aspect(1.0); _strip_ticks(ax)
            if r == 0:
                ax.set_title(angle_labels[col], pad=3, fontsize=8)
            if col == 0 and key in ("illum_ph", "out_amp", "rxk_amp"):
                _add_scalebar(ax, resolution_um, scalebar_length)
            last_im[key] = im

    def _phase_cbar(cax, im):
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        return cb
    for r, (key, lbl, cmap, vmn, vmx, pt) in enumerate(rows):
        cax = fig.add_subplot(gs[r, n])
        if pt:
            _phase_cbar(cax, last_im[key])
        else:
            cb = fig.colorbar(last_im[key], cax=cax); cb.ax.tick_params(labelsize=7)
    row_h = (TOP - BOTTOM) / n_rows
    for r, (key, lbl, *_rest) in enumerate(rows):
        y = TOP - row_h * (r + 0.5)
        fig.text(0.02, y, lbl, fontsize=8, rotation=90, va="center", ha="center",
                 fontweight="bold")
    return _save(fig, out_dir, "fig3d_angular_transmission_fields", formats)


def panel_4_multi_depth_xy(*, ss_opt_mat: Path, ss_opt_key: str = "ss_opt",
                           z_um=(15.0, 20.0, 25.0), resolution_um: float = 0.2581,
                           dz_um=None, scalebar_length: float = 10.0, cmap: str = "gray",
                           vmin=None, vmax=None, outline_colors=None, outline_lw: float = 2.5,
                           flip_z: bool = False, true_depth_focus_um=None,
                           true_depth_ref_idx=None, out_dir: Path, formats: Iterable[str]):
    """Fig 4 (multi-depth XY) — K XY cross-sections of the ODT volume.

    Grayscale by default with z = 15/20/25 μm. Each column carries a
    per-depth coloured frame + z title.

    Two z-axis conventions:
      - default (``true_depth_focus_um`` None): labels are VOLUME coordinates,
        z = index·dz measured from the BPM slab's front face. Index of a
        requested z = round(z / dz).
      - true-depth anchor (``true_depth_focus_um`` set): labels are TRUE sample
        depth. The BPM centres the zero-defocus plane at slice
        ``true_depth_ref_idx`` (default Nz//2+1, the DFR_Ori convention), which
        physically sits at ``true_depth_focus_um`` (the swept object-layer
        position). Then true_z(index) = focus + (index − ref)·dz, and a
        requested true z maps to index = round(ref + (z − focus)/dz)."""
    import matplotlib.pyplot as plt
    try:
        vol = _load_ss_opt_volume(ss_opt_mat, ss_opt_key)
    except FileNotFoundError as e:
        LOG.warning("panel_4_multi_depth_xy: %s; skipping.", e)
        return
    if not flip_z:
        vol = np.flip(vol, axis=0).copy()   # z increases with index (solver frame)
    Nz, Ny, Nx = vol.shape
    dz = dz_um if dz_um is not None else resolution_um
    if true_depth_focus_um is not None:
        ref = int(true_depth_ref_idx) if true_depth_ref_idx is not None else (Nz // 2 + 1)
        z_indices = [max(0, min(Nz - 1, int(round(ref + (z - true_depth_focus_um) / dz))))
                     for z in z_um]
        def _z_label(zi):
            return f"z = {true_depth_focus_um + (zi - ref) * dz:.1f} μm"
        LOG.info("panel_4_multi_depth_xy: TRUE-depth anchor focus=%.2fμm@idx%d dz=%.4f "
                 "z_um=%s -> idx=%s cmap=%s", true_depth_focus_um, ref, dz,
                 list(z_um), z_indices, cmap)
    else:
        z_indices = [max(0, min(Nz - 1, int(round(z / dz)))) for z in z_um]
        def _z_label(zi):
            return f"z = {zi * dz:.1f} μm"
        LOG.info("panel_4_multi_depth_xy: volume-coord vol %s dz=%.4f z_um=%s -> idx=%s cmap=%s",
                 tuple(vol.shape), dz, list(z_um), z_indices, cmap)
    K = len(z_indices)

    rvmin = vol.min() if vmin is None else vmin
    rvmax = float(vol.max() * 0.95) if vmax is None else vmax

    fig_w = 6.85
    LEFT, RIGHT, TOP, BOTTOM = 0.06, 0.98, 0.95, 0.04
    img_col_w_frac = (RIGHT - LEFT) / (K + 0.05)
    row_h_frac = (TOP - BOTTOM)
    fig = plt.figure(figsize=(fig_w, fig_w * img_col_w_frac / row_h_frac))
    gs = fig.add_gridspec(1, K + 1, width_ratios=[1.0] * K + [0.05],
                          hspace=0.06, wspace=0.04,
                          left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    outline_colors = _resolve_outline_palette(outline_colors, K)

    last_im = None
    for c, zi in enumerate(z_indices):
        ax = fig.add_subplot(gs[0, c])
        last_im = ax.imshow(vol[zi, :, :], cmap=cmap, vmin=rvmin, vmax=rvmax)
        ax.set_box_aspect(1.0); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(outline_colors[c]); sp.set_linewidth(outline_lw)
        ax.set_title(_z_label(zi), pad=4,
                     color=outline_colors[c], fontweight="bold")
        if c == 0:
            _add_scalebar(ax, resolution_um, scalebar_length)
    if last_im is not None:
        cb = fig.colorbar(last_im, cax=fig.add_subplot(gs[0, K]), label=r"$n - n_0$")
        cb.ax.tick_params(labelsize=7)
    fig.text(0.015, (TOP + BOTTOM) / 2, "BPM-ODT (GVAS)", fontsize=9, rotation=90,
             va="center", ha="center", fontweight="bold")
    return _save(fig, out_dir, "fig4f_multi_depth_xy_odt_images", formats)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_manuscript_panels(*, out_dir, rrmat=None, layer_dir=None, layer_epoch=1,
                             scan_stack=None, odt_input=None, ss_opt=None,
                             ss_opt_key="ss_opt",
                             resolution_um=0.2581, scalebar_um=10.0, dz_um=None,
                             wavelength_um=0.517, medium_n=1.33,
                             object_size_1b=120, crop_2d=None, angle_indices=None,
                             layer_positions=None,
                             fig3d_rrmat=None, fig3d_rrmat_propagate_d=None,
                             fig3d_rrmat_NA=None, fig3d_show_output_amp=True,
                             fig3d_subtract_output_ramp=False,
                             fig4_z_um=(15.0, 20.0, 25.0), fig4_cmap="gray",
                             fig4_true_depth_focus_um=None, fig4_true_depth_ref_idx=None,
                             fig4_vmin=None, fig4_vmax=None, fig4_outline_color=None,
                             formats=("png", "svg")) -> dict:
    """Render the six manuscript panels from pipeline artifacts.

    Each panel is guarded: a missing input or a failing panel is
    warned-and-skipped so the rest still render. Returns {stem: [paths]}.
    """
    import matplotlib
    matplotlib.use("Agg")
    _apply_style()
    out_dir = Path(out_dir)
    common = dict(out_dir=out_dir, formats=list(formats),
                  resolution_um=resolution_um, scalebar_length=scalebar_um)
    written: dict = {}

    def _run(stem, fn, **kw):
        try:
            res = fn(**kw)
            if res:
                written[stem] = res
        except Exception as e:
            LOG.error("panel %s failed: %s", stem, e)

    if rrmat is not None:
        _run("fig2c_confocal_reflectance", panel_1b, rrmat=rrmat,
             object_size=object_size_1b, wavelength_um=wavelength_um,
             medium_n=medium_n, **common)
    if layer_dir is not None:
        _run("fig2e_optimized_layers", panel_2c, layer_dir=layer_dir,
             epoch=layer_epoch, depths_um=layer_positions, **common)
        _run("fig2e_optimized_object", panel_2d, layer_dir=layer_dir,
             epoch=layer_epoch, crop_size=crop_2d, **common)
    if scan_stack is not None:
        _run("fig3b_depth_scanned_layers", panel_3b, scan_stack=scan_stack, **common)
    if odt_input is not None:
        _run("fig3d_angular_transmission_fields", panel_3d, odt_input_mat=odt_input,
             angle_indices=angle_indices, wavelength_um=wavelength_um,
             medium_n=medium_n, show_output_amp=fig3d_show_output_amp,
             subtract_output_ramp=fig3d_subtract_output_ramp,
             rrmat_path=fig3d_rrmat,
             rrmat_propagate_d=fig3d_rrmat_propagate_d, rrmat_NA=fig3d_rrmat_NA,
             **common)
    if ss_opt is not None:
        _run("fig4f_multi_depth_xy_odt_images", panel_4_multi_depth_xy,
             ss_opt_mat=ss_opt, ss_opt_key=ss_opt_key, z_um=fig4_z_um,
             cmap=fig4_cmap, dz_um=dz_um,
             true_depth_focus_um=fig4_true_depth_focus_um,
             true_depth_ref_idx=fig4_true_depth_ref_idx,
             vmin=fig4_vmin, vmax=fig4_vmax,
             outline_colors=([fig4_outline_color] if fig4_outline_color else None),
             **common)
    return written
