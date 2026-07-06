"""
multilayer_torch/scan_postprocess.py
-------------------------------------
Post-processing for depth-scan outputs from solver.py is_scan loop.

Collects scanning-layer reconstructions across all scan points, stacks
them into 3D numpy arrays, and renders a 2x2 video showing how amplitude
and phase of the input/output scattering layers vary with depth.

Public API
----------
    make_scan_postprocess(scan_dir, epoch=20, fps=2)
        Run the full pipeline: collect -> save .npy -> render video.
        Used by both cli.py 'scan-postprocess' subcommand and solver.py
        is_scan loop (auto-call after scan completes).

Outputs (saved next to scan_dir)
-------------------------------
    input_layer_stack.npy   shape (N_depths, H, W) complex64
    output_layer_stack.npy  shape (N_depths, H, W) complex64
    depths.npy              shape (N_depths,) float — z position per point
    scan_video.mp4          2x2 video (in/out x amp/phase) over depth
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

logger = logging.getLogger(__name__)

# Match scan_dz+2.00_z9.00 or scan_dz-3.00_z4.00
_SCAN_DIR_PATTERN = re.compile(r'scan_dz([+-]?\d+\.\d+)_z(\d+\.\d+)')


def _parse_scan_dirs(scan_root: Path) -> list[tuple[float, Path]]:
    """Return list of (depth_z, dir_path) sorted by depth."""
    entries = []
    for d in scan_root.iterdir():
        if not d.is_dir():
            continue
        m = _SCAN_DIR_PATTERN.match(d.name)
        if m:
            depth_z = float(m.group(2))
            entries.append((depth_z, d))
    entries.sort(key=lambda x: x[0])
    return entries


def _center_crop(img: np.ndarray, target: int) -> np.ndarray:
    """Centre-crop a 2D array to target x target."""
    h, w = img.shape
    r0 = h // 2 - target // 2
    c0 = w // 2 - target // 2
    return img[r0:r0 + target, c0:c0 + target]


def _load_scanning_layer(
    scan_subdir: Path, epoch: int, layer_index: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load input and output files for the specified layer index.

    Naming convention: layer index i -> file suffix _{i+1}_ (1-indexed).
    In reflection geometry, sample layer (index N-1) has only output file,
    no input file. Returns None if the output file doesn't exist (incomplete
    scan point — optimization was interrupted before saving checkpoints).
    """
    suffix = layer_index + 1  # 1-indexed
    fin = scan_subdir / f'input_layer_{suffix}_epoch_{epoch}.txt'
    fout = scan_subdir / f'output_layer_{suffix}_epoch_{epoch}.txt'
    if not fout.exists():
        return None
    out_arr = np.loadtxt(str(fout), dtype=np.complex64)
    # Input file may not exist for sample plane (last layer in reflection geom)
    if fin.exists():
        in_arr = np.loadtxt(str(fin), dtype=np.complex64)
    else:
        in_arr = out_arr  # fallback: use output as input duplicate
        logger.warning(
            f'No input_layer_{suffix} in {scan_subdir} (sample plane?); '
            f'using output_layer_{suffix} for both in video')
    return in_arr, out_arr


def _normalize_per_frame(layer: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide a complex layer by its max amplitude so |E| in [0, 1]."""
    mx = np.max(np.abs(layer))
    return layer / (mx + eps)


def _render_video(
    in_stack: np.ndarray,
    out_stack: np.ndarray,
    depths: np.ndarray,
    out_path: Path,
    fps: int = 2,
    phase_cmap: str = 'jet',
    pixel_size_um: float | None = None,
) -> Path:
    """Render 2x2 video: amp top, phase bottom; input left, output right.

    Each frame is normalized by its own max(|E|) so amplitude is in [0, 1].
    Phase row uses the full [-pi, pi] range. One shared colorbar per row.
    If pixel_size_um is provided, a scalebar is added to each panel.

    Returns the actual saved path (mp4 if ffmpeg available, gif otherwise).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor='white')
    fig.subplots_adjust(top=0.92, right=0.88, wspace=0.05, hspace=0.15)

    in0 = _normalize_per_frame(in_stack[0])
    out0 = _normalize_per_frame(out_stack[0])

    im_in_amp = axes[0, 0].imshow(np.abs(in0), cmap='hot', vmin=0, vmax=1)
    im_out_amp = axes[0, 1].imshow(np.abs(out0), cmap='hot', vmin=0, vmax=1)
    im_in_phase = axes[1, 0].imshow(np.angle(in0), cmap=phase_cmap,
                                    vmin=-np.pi, vmax=np.pi)
    im_out_phase = axes[1, 1].imshow(np.angle(out0), cmap=phase_cmap,
                                     vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title('Input |E| (normalized)')
    axes[0, 1].set_title('Output |E| (normalized)')
    axes[1, 0].set_title('Input arg(E)')
    axes[1, 1].set_title('Output arg(E)')
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Scalebar overlay (static across frames) — uses the helper from plotting.py
    if pixel_size_um is not None and pixel_size_um > 0:
        try:
            from .plotting import _add_scalebar
            fov_um = in_stack.shape[-1] * pixel_size_um
            # Round to a nice 5 um multiple, ~20% of FOV
            bar_len_um = max(1.0, round(fov_um / 5 / 5) * 5)
            for ax in axes.flat:
                _add_scalebar(ax, pixel_size_um, bar_len_um)
        except Exception as exc:
            logger.warning('scalebar overlay failed: %s', exc)

    # One shared colorbar per row (aligned with subplot row extents).
    # Bottom row subplots occupy approx y=[0.11, 0.46]; top row y=[0.54, 0.88].
    cbar_amp_ax = fig.add_axes([0.90, 0.56, 0.02, 0.34])
    cbar_amp = fig.colorbar(im_out_amp, cax=cbar_amp_ax)
    cbar_amp.set_label('|E| (per-frame normalized)', fontsize=10)

    cbar_phase_ax = fig.add_axes([0.90, 0.13, 0.02, 0.34])
    cbar_phase = fig.colorbar(im_out_phase, cax=cbar_phase_ax)
    cbar_phase.set_label('arg(E) [rad]', fontsize=10)
    cbar_phase.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar_phase.set_ticklabels(
        [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    title = fig.suptitle(f'Scattering layer at z = {depths[0]:.2f} um',
                         fontsize=14)

    def update(frame):
        in_n = _normalize_per_frame(in_stack[frame])
        out_n = _normalize_per_frame(out_stack[frame])
        im_in_amp.set_data(np.abs(in_n))
        im_out_amp.set_data(np.abs(out_n))
        im_in_phase.set_data(np.angle(in_n))
        im_out_phase.set_data(np.angle(out_n))
        title.set_text(f'Scattering layer at z = {depths[frame]:.2f} um')
        return im_in_amp, im_out_amp, im_in_phase, im_out_phase, title

    n = len(depths)
    ani = animation.FuncAnimation(
        fig, update, frames=n, blit=False, interval=1000 // fps)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        ani.save(str(out_path), writer=writer, dpi=120)
        saved = out_path
    except Exception as exc:
        # Fallback: GIF if ffmpeg unavailable
        gif_path = out_path.with_suffix('.gif')
        ani.save(str(gif_path), writer='pillow', fps=fps)
        logger.warning('ffmpeg unavailable (%s); saved %s instead', exc, gif_path)
        saved = gif_path

    plt.close(fig)
    return saved


def make_scan_postprocess(
    scan_dir: str | Path,
    epoch: int = 20,
    fps: int = 2,
    phase_cmap: str = 'jet',
    pixel_size_um: float | None = None,
    scanning_layer_index: int = 0,
) -> dict:
    """Collect scanned layers, save 3D stacks, and render video.

    Crop size is fixed to half of the maximum padded size found across
    input and output layers (matches the user's preferred crop choice).

    Parameters
    ----------
    scan_dir : path
        Directory containing scan_dz*_z* subdirectories.
    epoch : int
        Epoch number of checkpoints to load (default: 20).
    fps : int
        Video frame rate (default: 2).
    phase_cmap : str
        Colormap for phase panels (default: 'jet'). Common alternatives:
        'hsv' (cyclic), 'twilight' (cyclic, perceptually uniform).
    pixel_size_um : float, optional
        Pixel size of the scanning layer in micrometers. If provided, a
        scalebar is overlaid on each video panel.
    scanning_layer_index : int
        0-indexed layer that was scanned (default: 0). This is the layer
        whose position was swept — it changes across scan points, while
        other layers stay at their preoptimized values.

    Returns
    -------
    dict with keys:
        'in_stack_path', 'out_stack_path', 'depths_path', 'video_path',
        'in_stack_shape', 'out_stack_shape', 'n_depths'
    """
    scan_root = Path(scan_dir)
    if not scan_root.is_dir():
        raise NotADirectoryError(f'{scan_root} is not a directory')

    entries = _parse_scan_dirs(scan_root)
    if not entries:
        raise ValueError(f'No scan_* subdirectories found in {scan_root}')

    logger.info(
        'Found %d scan points in %s (scanning_layer_index=%d)',
        len(entries), scan_root, scanning_layer_index,
    )

    # Filter to only complete scan points (have checkpoint at requested epoch).
    # Load first valid entry to determine padded sizes.
    valid_entries = []
    crop_size = None
    for z, d in entries:
        result = _load_scanning_layer(d, epoch, scanning_layer_index)
        if result is None:
            logger.warning('Skipping incomplete scan point z=%.2f (%s)', z, d.name)
            continue
        if crop_size is None:
            in0, out0 = result
            crop_size = max(in0.shape[0], in0.shape[1],
                            out0.shape[0], out0.shape[1]) // 2
            logger.info(
                'Input raw=%dx%d, output raw=%dx%d, crop_size=%d (half of max)',
                in0.shape[0], in0.shape[1], out0.shape[0], out0.shape[1], crop_size,
            )
        valid_entries.append((z, d))

    if not valid_entries:
        raise ValueError(f'No complete scan points found at epoch {epoch}')
    if len(valid_entries) < len(entries):
        logger.info(
            '%d/%d scan points complete', len(valid_entries), len(entries))

    # Stack
    n = len(valid_entries)
    in_stack = np.zeros((n, crop_size, crop_size), dtype=np.complex64)
    out_stack = np.zeros((n, crop_size, crop_size), dtype=np.complex64)
    depths = np.zeros(n, dtype=np.float32)
    for i, (z, d) in enumerate(valid_entries):
        in_arr, out_arr = _load_scanning_layer(d, epoch, scanning_layer_index)
        in_stack[i] = _center_crop(in_arr, crop_size)
        out_stack[i] = _center_crop(out_arr, crop_size)
        depths[i] = z

    # Save arrays
    in_path = scan_root / 'input_layer_stack.npy'
    out_path = scan_root / 'output_layer_stack.npy'
    depths_path = scan_root / 'depths.npy'
    np.save(str(in_path), in_stack)
    np.save(str(out_path), out_stack)
    np.save(str(depths_path), depths)
    logger.info('Saved %s shape %s', in_path.name, in_stack.shape)
    logger.info('Saved %s shape %s', out_path.name, out_stack.shape)
    logger.info('Saved %s shape %s', depths_path.name, depths.shape)

    # Render video
    video_path = _render_video(
        in_stack, out_stack, depths,
        scan_root / 'scan_video.mp4', fps=fps, phase_cmap=phase_cmap,
        pixel_size_um=pixel_size_um,
    )
    logger.info('Saved %s', video_path)

    return {
        'in_stack_path': str(in_path),
        'out_stack_path': str(out_path),
        'depths_path': str(depths_path),
        'video_path': str(video_path),
        'in_stack_shape': in_stack.shape,
        'out_stack_shape': out_stack.shape,
        'n_depths': n,
    }
