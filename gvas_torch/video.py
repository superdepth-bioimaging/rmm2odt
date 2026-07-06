"""
video.py
========
Optional MP4 assembly from per-epoch PNG snapshots and latent field videos.

Uses imageio-ffmpeg; import is deferred to the function body so that
the rest of the package does not depend on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def images_to_mp4(
    image_paths: Iterable[Path],
    out_path: Path,
    fps: int = 5,
) -> None:
    """Encode a sequence of PNG frames into an MP4 file.

    Args:
        image_paths: ordered iterable of PNG paths.
        out_path:    destination MP4 path.
        fps:         frames per second.
    """
    import imageio.v2 as imageio  # local import — avoid eager dep

    frames = [imageio.imread(str(p)) for p in image_paths]
    if not frames:
        raise ValueError("No frames provided.")

    writer = imageio.get_writer(str(out_path), format="FFMPEG",
                                fps=fps, codec="libx264", quality=8)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


def _center_crop_2d(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def save_latent_video(
    U: np.ndarray,
    out_path: Path,
    crop_size: int | None = None,
    fps: int = 10,
) -> None:
    """Render all angles of the complex latent field as an MP4.

    Each frame shows amplitude (left) and phase (right) for one angle,
    centre-cropped to ``crop_size``.

    Args:
        U:         complex array of shape ``(n_angles, H, W)``.
        out_path:  destination MP4 path.
        crop_size: centre crop before rendering; ``None`` = no crop.
        fps:       frames per second.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from io import BytesIO

    n_angles = U.shape[0]

    # Pre-compute global ranges for consistent colormap scaling
    if crop_size is not None:
        abs_all = np.stack([np.abs(_center_crop_2d(U[i], crop_size)) for i in range(n_angles)])
    else:
        abs_all = np.abs(U)
    vmin_abs, vmax_abs = float(abs_all.min()), float(abs_all.max())

    writer = imageio.get_writer(str(out_path), format="FFMPEG",
                                fps=fps, codec="libx264", quality=8)
    try:
        for i in range(n_angles):
            slc = U[i]
            if crop_size is not None:
                slc = _center_crop_2d(slc, crop_size)

            fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
            im0 = axes[0].imshow(np.abs(slc), cmap="hot", vmin=vmin_abs, vmax=vmax_abs)
            axes[0].set_title(f"|U| (angle {i})")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)
            im1 = axes[1].imshow(np.angle(slc), cmap="jet", vmin=-np.pi, vmax=np.pi)
            axes[1].set_title(f"phase (angle {i})")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            frame = imageio.imread(buf, format="png")
            writer.append_data(frame)
    finally:
        writer.close()
