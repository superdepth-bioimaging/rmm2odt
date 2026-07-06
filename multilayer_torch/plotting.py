"""
multilayer_torch/plotting.py
-----------------------------
Scientific figure helpers matching the TF1.x plotting style exactly.

Plot types
----------
  save_initial_layers_figure  — N_layers x 4 grid (before optimization)
  save_confocal_figure        — 1 x 2 (amplitude | phase), called once
  save_layers_figure          — 2 x N per-epoch scattering layers
  save_focus_figure           — 2 x 2 per-epoch focus-plane object (z=0)
  save_pupils_figure          — 2 x 2 per-epoch pupil aberrations
  save_loss_figure            — loss curve, called per epoch
  save_global_phase_figure    — single-panel global phase map, per epoch
  save_correction_figure      — 2 x 3 post-correction comparison
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

_PHASE_TICKS = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
_PHASE_LABELS = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_colorbar(fig, ax, im, text_color: str = 'black'):
    """Attach a compact colorbar to the right of ax using make_axes_locatable."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.07)
    cbar = fig.colorbar(im, cax=cax)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontsize(9)
        lbl.set_color(text_color)
    cbar.ax.tick_params(colors=text_color)
    return cbar


def _add_scalebar(
    ax,
    pixel_size_um: float,
    length_um: float,
    bar_color: str = 'white',
    bg_color: str = 'black',
) -> None:
    """Overlay a scalebar in the lower-right corner of ax.

    Uses matplotlib_scalebar if installed; otherwise falls back to a
    simple line + text drawn in normalised axes coordinates.
    """
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(
            pixel_size_um, 'um',
            location='lower right', frameon=True,
            color=bar_color, box_color=bg_color, box_alpha=0.1,
            scale_loc='top', width_fraction=0.04,
            fixed_value=length_um,
            font_properties={'size': 10, 'weight': 'bold'},
        ))
        return
    except ImportError:
        pass

    # Fallback: draw line + label in axes-fraction coordinates
    xlim = ax.get_xlim()
    img_w = abs(xlim[1] - xlim[0])          # image width in data pixels
    bar_frac = (length_um / pixel_size_um) / img_w
    bar_frac = min(bar_frac, 0.40)           # never wider than 40 % of image

    x1 = 0.95
    x0 = x1 - bar_frac
    y = 0.07

    ax.plot([x0, x1], [y, y], color=bar_color, linewidth=3,
            solid_capstyle='butt', transform=ax.transAxes, clip_on=True)
    ax.text(
        (x0 + x1) / 2, y + 0.02, f'{length_um:.0f} \u03bcm',
        color=bar_color, ha='center', va='bottom',
        fontsize=9, fontweight='bold', transform=ax.transAxes,
    )


def _funcs_to_complex(arr: np.ndarray, ctype: str) -> np.ndarray:
    """Convert a 2-channel [2, H, W] float32 initializer array to complex [H, W]."""
    if ctype == 'Rectangular':
        return (arr[0] + 1j * arr[1]).astype(np.complex64)
    elif ctype == 'Euler':
        return (arr[0] * np.exp(1j * arr[1])).astype(np.complex64)
    else:  # phase-only complex
        return np.exp(1j * arr[1]).astype(np.complex64)


# ---------------------------------------------------------------------------
# 1.  Initial layers  (N_layers × 4 panel, before optimization starts)
# ---------------------------------------------------------------------------

def save_initial_layers_figure(
    input_funcs: list[np.ndarray],
    output_funcs: list[np.ndarray],
    complex_tensor_type: str,
    output_path: str | Path,
) -> None:
    """Save all initial scattering layers as a N_layers × 4 grid.

    Columns: [Input amp | Input phase | Output amp | Output phase]
    Colormaps: amplitude -> hot, phase -> jet (-pi .. +pi).
    Called once before optimization.

    Parameters
    ----------
    input_funcs, output_funcs : list of [2, H, W] float32 arrays
        One entry per layer (N_layers total, including sample plane).
    complex_tensor_type : 'Rectangular' | 'Euler' | 'phase-only complex'
    output_path : directory in which to write initial_layers.png
    """
    output_path = Path(output_path)
    n = len(output_funcs)

    in_c = [_funcs_to_complex(f, complex_tensor_type) for f in input_funcs]
    out_c = [_funcs_to_complex(f, complex_tensor_type) for f in output_funcs]

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), facecolor='white')
    fig.patch.set_facecolor('white')
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        inp = in_c[i] if i < len(in_c) else out_c[i]
        out = out_c[i]

        # col 0 – input amplitude
        im = axes[i, 0].imshow(np.abs(inp), cmap='hot')
        axes[i, 0].set_title(f'Input amp  L{i + 1}', color='black')
        cb = fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        # col 1 – input phase
        im = axes[i, 1].imshow(np.angle(inp), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[i, 1].set_title(f'Input phase  L{i + 1}', color='black')
        cb = fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        cb.set_ticks(_PHASE_TICKS)
        cb.set_ticklabels(_PHASE_LABELS)
        cb.ax.tick_params(labelsize=7)

        # col 2 – output amplitude
        im = axes[i, 2].imshow(np.abs(out), cmap='hot')
        axes[i, 2].set_title(f'Output amp  L{i + 1}', color='black')
        cb = fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        # col 3 – output phase
        im = axes[i, 3].imshow(np.angle(out), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axes[i, 3].set_title(f'Output phase  L{i + 1}', color='black')
        cb = fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        cb.set_ticks(_PHASE_TICKS)
        cb.set_ticklabels(_PHASE_LABELS)
        cb.ax.tick_params(labelsize=7)

        for ax in axes[i]:
            ax.tick_params(labelbottom=False, labelleft=False, length=0)

    fig.suptitle('Initial scattering layers', fontsize=14, color='black')
    fig.tight_layout()
    save_p = output_path / 'initial_layers.png'
    fig.savefig(str(save_p), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_p)


# ---------------------------------------------------------------------------
# 2.  Confocal image  (1 × 2: amplitude | phase)
# ---------------------------------------------------------------------------

def save_confocal_figure(
    confocal_image: np.ndarray,
    resolution: float,
    save_path: str | Path,
    scanning_size: int | None = None,
    dpi: int = 300,
) -> None:
    """Save the confocal reference image as a 1 × 2 figure (amplitude | phase).

    Called once after data loading / preprocessing.

    Parameters
    ----------
    confocal_image : 2-D complex ndarray  [H, W]
    resolution     : µm / pixel (used for scalebar)
    save_path      : output file path
    scanning_size  : shown in title if provided
    """
    save_path = Path(save_path)
    sz = confocal_image.shape[-1]
    fov_um = resolution * sz
    scalebar_len = max(1.0, round(fov_um / 8 / 5) * 5)

    size_info = f'  |  {sz}x{sz} px  |  FOV = {fov_um:.1f} um'
    if scanning_size is not None:
        size_info = f'  |  scanning = {scanning_size} px' + size_info

    amp = np.abs(confocal_image)
    phase = np.angle(confocal_image)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi, facecolor='white')
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'Confocal Reference Image{size_info}',
        color='black', fontsize=11, fontweight='bold', y=1.02,
    )

    # Amplitude
    im_a = axes[0].imshow(amp, cmap='hot', interpolation='nearest')
    axes[0].set_title('Amplitude  |amp|', color='black', fontsize=10, pad=5)
    axes[0].axis('off')
    _add_colorbar(fig, axes[0], im_a, text_color='black')
    _add_scalebar(axes[0], resolution, scalebar_len)

    # Phase
    im_p = axes[1].imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi,
                          interpolation='nearest')
    axes[1].set_title('Phase  angle', color='black', fontsize=10, pad=5)
    axes[1].axis('off')
    cbar_p = _add_colorbar(fig, axes[1], im_p, text_color='black')
    cbar_p.set_ticks(_PHASE_TICKS)
    cbar_p.set_ticklabels(_PHASE_LABELS)
    _add_scalebar(axes[1], resolution, scalebar_len)

    fig.subplots_adjust(wspace=0.25, left=0.04, right=0.96, top=0.92, bottom=0.04)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_path)


# ---------------------------------------------------------------------------
# 3.  Scattering layers per epoch  (2 × N: amplitude row | phase row)
# ---------------------------------------------------------------------------

def save_layers_figure(
    layers: list[np.ndarray],
    layer_positions: list[float] | np.ndarray,
    resolutions: float | list[float],
    save_path: str | Path,
    title: str = '',
    epoch: int | None = None,
    dpi: int = 300,
) -> None:
    """Save scattering layers as a 2 × N_layers scientific figure.

    Each layer is centre-cropped to half its original size so deeper
    (larger-padded) layers are shown at their natural central FOV.

    Parameters
    ----------
    layers         : list of complex [H, W] ndarrays (N_layers entries)
    layer_positions: axial depths in µm (deepest first, ending at 0)
    resolutions    : µm / pixel — scalar or one value per layer
    save_path      : output file path (PDF recommended)
    title          : figure title prefix ('Input Layers' or 'Output Layers')
    epoch          : if given, appended to title as 'Epoch N+1'
    """
    save_path = Path(save_path)
    n = len(layers)
    if n == 0:
        return

    if not isinstance(resolutions, (list, tuple, np.ndarray)):
        resolutions = [float(resolutions)] * n

    ep_label = f'  --  Epoch {epoch + 1}' if epoch is not None else ''
    fig_title = f'{title}{ep_label}'

    fig_w = max(3 * n, 6)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 6), dpi=dpi,
                             facecolor='white', constrained_layout=False)
    fig.patch.set_facecolor('white')
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(fig_title, color='black', fontsize=12, fontweight='bold', y=1.01)

    for i, layer in enumerate(layers):
        pix = float(resolutions[i])
        depth = float(layer_positions[i])

        # Centre-crop to half original size
        sz_orig = layer.shape[-1]
        sz_half = max(1, sz_orig // 2)
        margin = (sz_orig - sz_half) // 2
        crop = layer[margin: margin + sz_half, margin: margin + sz_half]

        fov_um = pix * sz_half
        scalebar_len = max(1.0, round(fov_um / 8 / 5) * 5)

        # Amplitude row
        ax_a = axes[0, i]
        im_a = ax_a.imshow(np.abs(crop), cmap='gray', interpolation='nearest')
        ax_a.set_title(f'z = {depth:.0f} um\nFOV = {fov_um:.0f} um',
                       color='black', fontsize=9, pad=3)
        ax_a.axis('off')
        _add_colorbar(fig, ax_a, im_a, text_color='black')

        # Phase row
        ax_p = axes[1, i]
        im_p = ax_p.imshow(np.angle(crop), cmap='jet', vmin=-np.pi, vmax=np.pi,
                           interpolation='nearest')
        ax_p.axis('off')
        cb_p = _add_colorbar(fig, ax_p, im_p, text_color='black')
        cb_p.set_ticks(_PHASE_TICKS)
        cb_p.set_ticklabels(_PHASE_LABELS)
        _add_scalebar(ax_p, pix, scalebar_len)

    # Row labels on leftmost column
    for row_ax, row_label in zip(
        [axes[0, 0], axes[1, 0]], ['Amplitude', 'Phase [rad]']
    ):
        row_ax.annotate(
            row_label,
            xy=(-0.22, 0.5), xycoords='axes fraction',
            rotation=90, va='center', ha='center',
            color='black', fontsize=10, fontweight='bold',
        )

    fig.subplots_adjust(wspace=0.08, hspace=0.15, left=0.12)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_path)


# ---------------------------------------------------------------------------
# 4.  Focus-plane object per epoch  (2 × 2: input / output channel)
# ---------------------------------------------------------------------------

def save_focus_figure(
    input_focus: np.ndarray,
    output_focus: np.ndarray,
    resolution: float,
    save_path: str | Path,
    epoch: int | None = None,
    dpi: int = 300,
) -> None:
    """Save focus-plane (z = 0) object as a 2 × 2 figure.

    Layout:
      Row 0 — amplitude:  [Input channel]  [Output channel]
      Row 1 — phase:      [Input channel]  [Output channel]

    Called per epoch.
    """
    save_path = Path(save_path)
    ep_label = f'  --  Epoch {epoch + 1}' if epoch is not None else ''
    fig_title = f'Focus Object (z = 0 um){ep_label}'

    sz = input_focus.shape[-1]
    fov_um = resolution * sz
    scalebar_len = max(1.0, round(fov_um / 8 / 5) * 5)

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=dpi,
                             facecolor='white', constrained_layout=False)
    fig.patch.set_facecolor('white')
    fig.suptitle(fig_title, color='black', fontsize=12, fontweight='bold', y=1.01)

    col_titles = ['Input channel', 'Output channel']
    for col, (focus, ct) in enumerate(zip([input_focus, output_focus], col_titles)):
        amp = np.abs(focus)
        phase = np.angle(focus)

        ax_a = axes[0, col]
        im_a = ax_a.imshow(amp, cmap='hot', interpolation='nearest')
        ax_a.set_title(f'{ct}\nFOV = {fov_um:.0f} um', color='black', fontsize=9, pad=3)
        ax_a.axis('off')
        _add_colorbar(fig, ax_a, im_a, text_color='black')

        ax_p = axes[1, col]
        im_p = ax_p.imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi,
                           interpolation='nearest')
        ax_p.axis('off')
        cb_p = _add_colorbar(fig, ax_p, im_p, text_color='black')
        cb_p.set_ticks(_PHASE_TICKS)
        cb_p.set_ticklabels(_PHASE_LABELS)
        _add_scalebar(ax_p, resolution, scalebar_len)

    for row_ax, row_label in zip(
        [axes[0, 0], axes[1, 0]], ['Amplitude', 'Phase [rad]']
    ):
        row_ax.annotate(
            row_label,
            xy=(-0.22, 0.5), xycoords='axes fraction',
            rotation=90, va='center', ha='center',
            color='black', fontsize=10, fontweight='bold',
        )

    fig.subplots_adjust(wspace=0.08, hspace=0.14, left=0.14)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_path)


# ---------------------------------------------------------------------------
# 4b.  Corrected confocal image (Model B)  —  S(r) = Σ_k I_k(r)
# ---------------------------------------------------------------------------

def save_corrected_confocal_figure(
    corrected_confocal: np.ndarray,
    resolution: float,
    save_path: str | Path,
    epoch: int | None = None,
    dpi: int = 300,
) -> None:
    """Save the coherent-sum corrected confocal image S(r) = Σ_k I_k(r)
    produced by Model B as a 1×2 (amplitude / phase) figure.

    Parameters
    ----------
    corrected_confocal : complex ndarray [H, W]  — S(r) at the sample plane.
    """
    save_path = Path(save_path)
    ep_label = f'  --  Epoch {epoch + 1}' if epoch is not None else ''
    fig_title = f'Corrected confocal  (S = Σ_k E_back · conj(E_in) / (|E_in|² + ε)){ep_label}'

    sz = corrected_confocal.shape[-1]
    fov_um = resolution * sz
    scalebar_len = max(1.0, round(fov_um / 8 / 5) * 5)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi,
                             facecolor='white', constrained_layout=False)
    fig.patch.set_facecolor('white')
    fig.suptitle(fig_title, color='black', fontsize=11, fontweight='bold', y=1.02)

    amp = np.abs(corrected_confocal)
    phase = np.angle(corrected_confocal)
    total_intensity = float((amp ** 2).sum())

    ax_a = axes[0]
    im_a = ax_a.imshow(amp, cmap='hot', interpolation='nearest')
    ax_a.set_title(
        f'|S(r)|   (FOV = {fov_um:.0f} um)   total |S|² = {total_intensity:.3e}',
        color='black', fontsize=9, pad=3,
    )
    ax_a.axis('off')
    _add_colorbar(fig, ax_a, im_a, text_color='black')

    ax_p = axes[1]
    im_p = ax_p.imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi,
                       interpolation='nearest')
    ax_p.set_title('angle(S(r))', color='black', fontsize=9, pad=3)
    ax_p.axis('off')
    cb_p = _add_colorbar(fig, ax_p, im_p, text_color='black')
    cb_p.set_ticks(_PHASE_TICKS)
    cb_p.set_ticklabels(_PHASE_LABELS)
    _add_scalebar(ax_p, resolution, scalebar_len)

    fig.subplots_adjust(wspace=0.10)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_path)


# ---------------------------------------------------------------------------
# 4c.  Intensity-history tracking plot (Model B)
# ---------------------------------------------------------------------------

def save_intensity_history_figure(
    intensity_history: list[float],
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """Save a per-epoch tracking plot of total intensity = -cost_B.

    cost_B is stored as a negative number (we minimise it). The
    physically meaningful quantity is total |Σ_k I_k|² = -cost_B,
    plotted on a log scale.
    """
    save_path = Path(save_path)
    epochs = list(range(1, len(intensity_history) + 1))
    total_int = [-x for x in intensity_history]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor='white', dpi=dpi)
    ax.plot(epochs, total_int, 'g-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(r'Total intensity  $\Sigma_r |\Sigma_k I_k(r)|^2$', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Corrected-confocal total intensity (Model B)', fontsize=12)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_path)


# ---------------------------------------------------------------------------
# 5.  Pupil aberrations per epoch  (2 × 2: input / output pupil)
# ---------------------------------------------------------------------------

def save_pupils_figure(
    input_pupil: np.ndarray,
    output_pupil: np.ndarray,
    pixel_size: float,
    save_path: str | Path,
    epoch: int | None = None,
    dpi: int = 150,
) -> None:
    """Save input and output pupil aberrations as a 2 × 2 figure.

    Layout:
      Row 0 — amplitude:  [Input Pupil]  [Output Pupil]
      Row 1 — phase:      [Input Pupil]  [Output Pupil]

    Called per epoch.
    """
    save_path = Path(save_path)
    ep_label = f'  --  Epoch {epoch + 1}' if epoch is not None else ''
    fig_title = f'Pupil Aberrations{ep_label}'

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=dpi,
                             facecolor='white', constrained_layout=False)
    fig.patch.set_facecolor('white')
    fig.suptitle(fig_title, color='black', fontsize=12, fontweight='bold', y=1.01)

    col_titles = ['Input Pupil', 'Output Pupil']
    for col, (pupil, ct) in enumerate(zip([input_pupil, output_pupil], col_titles)):
        amp = np.abs(pupil)
        phase = np.angle(pupil)

        ax_a = axes[0, col]
        im_a = ax_a.imshow(amp, cmap='gray', interpolation='nearest')
        ax_a.set_title(ct, color='black', fontsize=9, pad=3)
        ax_a.axis('off')
        _add_colorbar(fig, ax_a, im_a, text_color='black')

        ax_p = axes[1, col]
        im_p = ax_p.imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi,
                           interpolation='nearest')
        ax_p.axis('off')
        cb_p = _add_colorbar(fig, ax_p, im_p, text_color='black')
        cb_p.set_ticks(_PHASE_TICKS)
        cb_p.set_ticklabels(_PHASE_LABELS)

    for row_ax, row_label in zip(
        [axes[0, 0], axes[1, 0]], ['Amplitude', 'Phase [rad]']
    ):
        row_ax.annotate(
            row_label,
            xy=(-0.22, 0.5), xycoords='axes fraction',
            rotation=90, va='center', ha='center',
            color='black', fontsize=10, fontweight='bold',
        )

    fig.subplots_adjust(wspace=0.08, hspace=0.14, left=0.14)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_path)


# ---------------------------------------------------------------------------
# 6.  Loss curve  (twinx: Pearson left/red, L2 right/blue)
# ---------------------------------------------------------------------------

def save_loss_figure(
    pearson_history: list[float],
    reg_sum_history: list[float],
    output_dir: str | Path,
) -> None:
    """Save dual-axis loss figure: Pearson correlation + total regularization.

    Left  axis (red):  Pearson correlation
    Right axis (blue): Total weighted regularization (sum of all gamma*term)

    Parameters
    ----------
    pearson_history  : per-epoch Pearson correlation
    reg_sum_history  : per-epoch total weighted regularization
    output_dir       : directory to write loss.png
    """
    output_dir = Path(output_dir)
    epochs = list(range(1, len(pearson_history) + 1))

    fig, ax1 = plt.subplots(figsize=(7, 4), facecolor='white')
    fig.suptitle('Losses Over Epochs', fontsize=12, color='black')

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Pearson correlation', color='red', fontsize=11)
    line1, = ax1.plot(epochs, pearson_history, 'r-o', markersize=4,
                      linewidth=1.5, label='Pearson')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Regularization (sum)', color='blue', fontsize=11)
    line2, = ax2.plot(epochs, reg_sum_history[:len(epochs)], 'b-s', markersize=4,
                      linewidth=1.5, label='Reg sum')
    ax2.tick_params(axis='y', labelcolor='blue')

    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'loss.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.  Global phase map  (single panel, per epoch)
# ---------------------------------------------------------------------------

def save_global_phase_figure(
    phase_map: np.ndarray,
    epoch: int,
    output_dir: str | Path,
) -> None:
    """Save global phase map to global_phase_epoch_{N}.png. Called per epoch."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='white')
    im = ax.imshow(phase_map, cmap='jet', vmin=-np.pi, vmax=np.pi)
    ax.set_title(f'Global Phase Map  --  Epoch {epoch + 1}', color='black')
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(_PHASE_TICKS)
    cbar.set_ticklabels(_PHASE_LABELS)
    fig.tight_layout()
    save_p = output_dir / f'global_phase_epoch_{epoch + 1}.png'
    fig.savefig(str(save_p), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_p)


# ---------------------------------------------------------------------------
# 7b.  Per-source phase/amplitude correction  (1 × 2)
# ---------------------------------------------------------------------------

def save_source_correction_figure(
    source_phase: np.ndarray,
    source_log_amp: np.ndarray,
    scanning_size: int,
    epoch: int,
    output_dir: str | Path,
) -> None:
    """Save per-source phase and log-amplitude maps.

    Parameters
    ----------
    source_phase : 1-D array [N_sources]
    source_log_amp : 1-D array [N_sources]
    scanning_size : int — grid side for reshaping (maps are scanning_size x scanning_size)
    epoch : int (0-indexed)
    output_dir : output directory
    """
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor='white')

    # Reshape to 2D scanning grid
    N = len(source_phase)
    Nx = int(np.sqrt(N))
    # If scanning_size differs from sqrt(N), use scanning_size for display
    sz = min(scanning_size, Nx)
    phase_map = source_phase[:sz * sz].reshape(sz, sz)
    amp_map = source_log_amp[:sz * sz].reshape(sz, sz)

    # Source phase
    im0 = axes[0].imshow(phase_map, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title(f'Source Phase (epoch {epoch + 1})', color='black', fontsize=10)
    axes[0].axis('off')
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_ticks(_PHASE_TICKS)
    cbar0.set_ticklabels(_PHASE_LABELS)

    # Source log-amplitude
    im1 = axes[1].imshow(amp_map, cmap='RdBu_r')
    axes[1].set_title(f'Source Log-Amp (epoch {epoch + 1})', color='black', fontsize=10)
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_p = output_dir / f'source_correction_epoch_{epoch + 1}.png'
    fig.savefig(str(save_p), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_p)


# ---------------------------------------------------------------------------
# 8.  Post-correction comparison  (2 × 3)
# ---------------------------------------------------------------------------

def save_correction_figure(
    uncorrected: np.ndarray,
    corrected: np.ndarray,
    sample_layer: np.ndarray,
    output_dir: str | Path,
    resolution: float | None = None,
    scalebar_um: float = 10.0,
    dpi: int = 300,
) -> None:
    """Save correction comparison as a 2 × 3 figure.

    Layout:
      Row 0 — amplitude:  [Uncorrected]  [Corrected]  [Reconstructed object]
      Row 1 — phase:      [Uncorrected]  [Corrected]  [Reconstructed object]

    Called once after apply_scattering_correction().
    """
    output_dir = Path(output_dir)
    panels = [
        ('Uncorrected confocal', uncorrected),
        ('Corrected confocal',   corrected),
        ('Reconstructed object', sample_layer),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor='white')
    fig.patch.set_facecolor('white')

    for col, (title, img) in enumerate(panels):
        amp = np.abs(img)
        phase = np.angle(img)

        im_a = axes[0, col].imshow(amp, cmap='afmhot', interpolation='nearest')
        axes[0, col].set_title(title, fontsize=10, color='black')
        fig.colorbar(im_a, ax=axes[0, col], fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        if resolution is not None and scalebar_um and scalebar_um > 0:
            _add_scalebar(axes[0, col], resolution, scalebar_um)

        im_p = axes[1, col].imshow(
            phase, cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='nearest'
        )
        cb = fig.colorbar(im_p, ax=axes[1, col], fraction=0.046, pad=0.04)
        cb.set_ticks(_PHASE_TICKS)
        cb.set_ticklabels(_PHASE_LABELS)
        cb.ax.tick_params(labelsize=7)

        for ax in axes[:, col]:
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            ax.set_facecolor('white')

    axes[0, 0].set_ylabel('Amplitude', fontsize=10, color='black')
    axes[1, 0].set_ylabel('Phase', fontsize=10, color='black')

    fig.tight_layout()
    save_p = output_dir / 'confocal_corrected.png'
    fig.savefig(str(save_p), dpi=dpi, bbox_inches='tight', facecolor='white')
    save_svg = output_dir / 'confocal_corrected.svg'
    fig.savefig(str(save_svg), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info('Saved %s', save_p)
    logger.info('Saved %s', save_svg)


# ---------------------------------------------------------------------------
# 9.  E_focus preview  (3 × 3 grid of first 9 frames)
# ---------------------------------------------------------------------------

def save_efocus_preview(
    E_focus_all: np.ndarray,
    output_dir: str | Path,
    dpi: int = 300,
) -> None:
    """Save amplitude of the first 9 E_focus frames as a 3 × 3 grid.

    Called after apply_scattering_correction().
    """
    output_dir = Path(output_dir)
    n_prev = min(9, E_focus_all.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), facecolor='white')
    for idx, ax in enumerate(axes.ravel()):
        if idx < n_prev:
            ax.imshow(np.abs(E_focus_all[idx]), cmap='hot', interpolation='nearest')
            ax.set_title(f'E_focus[{idx}]', fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    save_p = output_dir / 'E_focus_preview.png'
    fig.savefig(str(save_p), dpi=dpi)
    plt.close(fig)
    logger.info('Saved %s', save_p)
