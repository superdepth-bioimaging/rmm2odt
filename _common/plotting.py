"""Minimal matplotlib helpers (pruned from research_projects/utils/plotting.py).

Only the functions the rmm2odt slice renderer needs: a compact colorbar and
a scalebar (with a pure-matplotlib fallback if matplotlib_scalebar is absent).
"""
from __future__ import annotations

import numpy as np

PHASE_TICKS = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
PHASE_LABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]


def add_colorbar(fig, ax, im, text_color: str = "black"):
    """Attach a compact colorbar to the right of `ax`."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    cbar = fig.colorbar(im, cax=cax)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontsize(9)
        lbl.set_color(text_color)
    cbar.ax.tick_params(colors=text_color)
    return cbar


def add_scalebar(ax, pixel_size_um: float, length_um: float,
                 bar_color: str = "white") -> None:
    """Overlay a scalebar. Uses matplotlib_scalebar if available, else a
    simple line + text in axes-fraction coordinates."""
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(
            pixel_size_um, "um", location="lower right", frameon=True,
            color=bar_color, box_color="black", box_alpha=0.1,
            scale_loc="top", width_fraction=0.04, fixed_value=length_um,
            font_properties={"size": 10, "weight": "bold"},
        ))
        return
    except ImportError:
        pass
    xlim = ax.get_xlim()
    img_w = abs(xlim[1] - xlim[0])
    bar_frac = min((length_um / pixel_size_um) / img_w, 0.40)
    x1 = 0.95
    x0 = x1 - bar_frac
    y = 0.07
    ax.plot([x0, x1], [y, y], color=bar_color, linewidth=3,
            solid_capstyle="butt", transform=ax.transAxes, clip_on=True)
    ax.text((x0 + x1) / 2, y + 0.02, f"{length_um:.0f} µm",
            color=bar_color, ha="center", va="bottom",
            fontsize=9, fontweight="bold", transform=ax.transAxes)
