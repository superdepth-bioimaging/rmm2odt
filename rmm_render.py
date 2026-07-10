"""Stage 4 — minimal ODT renderer: orthogonal RI slices + z-scan video.

Reproduces the rODT_vis / render_paper_panels display transform
(lateral fftshift -> Z-first transpose -> Z flip) so the orientation matches
the rest of the toolchain, then writes:
  - fig_ortho.{png,svg}   XY | XZ | YZ central slices
  - ri_zscan.mp4 (or .gif fallback)   sweep of XY planes through z

Matplotlib-only (no pyvista). Accepts either a ss_opt .mat path or an
in-memory (Ny, Nx, Nz) volume array.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _load_ss_opt(mat_path: str, key: str = "ss_opt") -> np.ndarray:
    """Load a 3-D RI volume (Ny, Nx, Nz) from a ss_opt .mat (v7 or v7.3)."""
    p = Path(mat_path)
    try:
        import scipy.io as sio
        m = sio.loadmat(str(p))
        if key not in m:
            avail = [k for k in m if not k.startswith("__")]
            key = avail[0] if avail else key
        return np.asarray(m[key])
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(str(p), "r") as f:
            if key not in f:
                key = [k for k in f.keys()][0]
            arr = np.array(f[key])
        # h5py reads HDF5 reversed vs MATLAB -> reverse all axes
        return np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))


def _display_transform(vol: np.ndarray) -> np.ndarray:
    """(Ny, Nx, Nz) on disk -> (Nz, Ny, Nx) for display.
    Lateral fftshift on axes (0,1), move Z to front, flip Z. Mirrors
    rODT_vis/core.py and render_paper_panels._load_ss_opt_volume."""
    if np.iscomplexobj(vol):
        vol = vol.real
    vol = np.fft.fftshift(np.fft.fftshift(vol, axes=1), axes=0)
    vol = np.transpose(vol, (2, 0, 1))      # (Ny,Nx,Nz) -> (Nz,Ny,Nx)
    vol = np.flip(vol, axis=0)
    return np.ascontiguousarray(vol)


def render_odt(volume, out_dir: str, *, resolution_um: float = 0.2581,
               dz_um: float | None = None, scalebar_um: float = 10.0,
               cmap: str = "inferno", ss_opt_key: str = "ss_opt",
               zscan_video: bool = True, make_ortho: bool = True,
               vmin=None, vmax=None) -> dict:
    """Render ortho slices (+ optional z-scan video) from a ss_opt volume.

    `volume` may be a path to a ss_opt .mat or an (Ny, Nx, Nz) ndarray.
    Returns dict of written file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _common.plotting import add_colorbar, add_scalebar

    if isinstance(volume, (str, os.PathLike)):
        vol = _load_ss_opt(str(volume), ss_opt_key)
    else:
        vol = np.asarray(volume)
    vol = _display_transform(vol)               # (Nz, Ny, Nx)
    Nz, Ny, Nx = vol.shape
    dz = dz_um if dz_um is not None else resolution_um
    if vmax is None:
        vmax = float(np.nanmax(vol))
    if vmin is None:
        vmin = float(np.nanmin(vol))

    os.makedirs(out_dir, exist_ok=True)
    out = {}

    # ---- ortho figure: XY (z mid) | XZ (y mid) | YZ (x mid) ----
    if make_ortho:
        xy = vol[Nz // 2, :, :]
        xz = vol[:, Ny // 2, :]                      # (Nz, Nx)
        yz = vol[:, :, Nx // 2]                      # (Nz, Ny)
        aspect_axial = dz / resolution_um
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
        im = axs[0].imshow(xy, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title("XY (central z)")
        add_scalebar(axs[0], resolution_um, scalebar_um)
        axs[1].imshow(xz, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_axial)
        axs[1].set_title("XZ (central y)")
        axs[2].imshow(yz, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_axial)
        axs[2].set_title("YZ (central x)")
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        add_colorbar(fig, axs[2], im)
        fig.suptitle("ODT reconstruction (n - n0)", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fp = os.path.join(out_dir, f"fig_ortho.{ext}")
            fig.savefig(fp, dpi=200, bbox_inches="tight")
            out[ext] = fp
        plt.close(fig)

    # ---- z-scan video ----
    if zscan_video:
        out["video"] = _zscan_video(vol, out_dir, cmap, vmin, vmax,
                                    resolution_um, scalebar_um)
    return out


def _zscan_video(vol, out_dir, cmap, vmin, vmax, resolution_um, scalebar_um):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    Nz = vol.shape[0]
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    frames = []
    for z in range(Nz):
        rgba = mapper.to_rgba(vol[z], bytes=True)[:, :, :3]
        frames.append(np.ascontiguousarray(rgba))
    mp4 = os.path.join(out_dir, "ri_zscan.mp4")
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(mp4, fps=8, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        return mp4
    except Exception:
        gif = os.path.join(out_dir, "ri_zscan.gif")
        try:
            import imageio.v2 as imageio
            imageio.mimsave(gif, frames, fps=8)
            return gif
        except Exception as e:
            return f"(z-scan video skipped: {e})"
