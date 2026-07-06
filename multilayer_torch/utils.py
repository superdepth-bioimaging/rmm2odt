"""
multilayer_torch/utils.py
--------------------------
Numpy-only preprocessing utilities for the multi-layer model.

No PyTorch imports. All functions operate on numpy arrays.
Called by solver.py (data loading / preprocessing) and correction.py.

Deliberate improvements over bioimaging_subfunctions.py originals:
  - butterworth_lowpass_filter: vectorised meshgrid replaces O(N²) Python loop.
  - generate_mask_indexs: plt.figure() / plt.imshow() side effects removed.
  - _pad_np / _crop_np: unified helpers used by all size-change operations.
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom as _scipy_zoom


# ---------------------------------------------------------------------------
# Centred FFT2 / IFFT2  (operate on [..., H, W] arrays)
# ---------------------------------------------------------------------------

def np_fft2(imgs: np.ndarray) -> np.ndarray:
    """Centred 2-D FFT over last two axes of *imgs*."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(imgs, axes=(-2, -1))),
        axes=(-2, -1),
    )


def np_ifft2(imgs: np.ndarray) -> np.ndarray:
    """Centred 2-D IFFT over last two axes of *imgs*."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(imgs, axes=(-2, -1))),
        axes=(-2, -1),
    )


# ---------------------------------------------------------------------------
# Padding / cropping  (match bioimaging_subfunctions.pad and crop_img)
# ---------------------------------------------------------------------------

def _pad_np(data: np.ndarray, target: int, mode: str = 'constant', **kwargs) -> np.ndarray:
    """Centre-pad spatial dims of a 2-D or 3-D [N, H, W] array to target × target.

    Convention matches bioimaging_subfunctions.pad:
        when (target − size) is odd the extra pixel goes on the top / left side.
    """
    m = data.shape[-1]
    half = (target - m) // 2
    extra = (target - m) % 2      # 1 if difference is odd, else 0
    if data.ndim == 2:
        pw = ((half + extra, half), (half + extra, half))
    else:  # ndim == 3  — matches TF1 symmetric 3-D behaviour
        pw = ((0, 0), (half, half), (half, half))
    return np.pad(data, pw, mode, **kwargs)


def _crop_np(data: np.ndarray, target: int) -> np.ndarray:
    """Centre-crop spatial dims of a 2-D or 3-D [N, H, W] array to target × target."""
    h, w = data.shape[-2], data.shape[-1]
    r0 = h // 2 - target // 2
    c0 = w // 2 - target // 2
    if data.ndim == 2:
        return data[r0:r0 + target, c0:c0 + target]
    return data[:, r0:r0 + target, c0:c0 + target]


def center_crop_np(data: np.ndarray, target_size: int) -> np.ndarray:
    """Centre-crop spatial dims of [N, H, W] or [H, W] array to target_size x target_size."""
    return _crop_np(data, target_size)


def change_size_np(data: np.ndarray, size_new: int, **kwargs) -> np.ndarray:
    """Crop or pad spatial dims of a 2-D or 3-D array to size_new × size_new."""
    m = data.shape[-1]
    if m < size_new:
        return _pad_np(data, size_new, **kwargs)
    if m > size_new:
        return _crop_np(data, size_new)
    return data


# ---------------------------------------------------------------------------
# Resolution / ROI matching
# ---------------------------------------------------------------------------

def _change_sampling_size(images: np.ndarray, new_size: int) -> np.ndarray:
    """Rescale [N, H, W] array to [N, new_size, new_size].

    Complex arrays are handled by zooming real and imaginary parts separately
    so that scipy.ndimage.zoom (which may not support complex) always works.
    """
    if images.shape[1] == new_size:
        return images
    factor = new_size / images.shape[1]
    zoom_factors = (1.0, factor, factor)
    if np.iscomplexobj(images):
        r = _scipy_zoom(images.real, zoom_factors, order=1)
        i = _scipy_zoom(images.imag, zoom_factors, order=1)
        return (r + 1j * i).astype(images.dtype)
    return _scipy_zoom(images, zoom_factors, order=1).astype(images.dtype)


def match_ROIandres(
    images: np.ndarray,
    current_resolution: float,
    new_resolution: float,
    new_p_size: int,
) -> np.ndarray:
    """Rescale images to *new_resolution* then crop / pad to *new_p_size* × *new_p_size*.

    Parameters
    ----------
    images             : [N, H, W] complex or real array
    current_resolution : pixel size of input images (same units as new_resolution)
    new_resolution     : desired pixel size
    new_p_size         : output spatial size in pixels
    """
    if current_resolution != new_resolution:
        image_length = images.shape[1] * current_resolution
        new_sampling_size = int(image_length // new_resolution)
        images = _change_sampling_size(images, new_sampling_size)
    return change_size_np(images, new_p_size)


# ---------------------------------------------------------------------------
# Mask index generation  (plt side effects removed)
# ---------------------------------------------------------------------------

def generate_mask_indexs(
    gm_extend_size: int,
    gm_size: int,
    gm_pad: int = 0,
    gm_num_sections: int = 1,
    gm_num_row: int = 0,
    gm_num_collumn: int = 0,
) -> np.ndarray:
    """Return flat (column-major) indices of the active mask region.

    Parameters
    ----------
    gm_extend_size  : total FOV size in pixels (square)
    gm_size         : size of each cropped ROI before padding
    gm_pad          : padding added to each cropped ROI
    gm_num_sections : number of sections (usually 1)
    gm_num_row      : row offset in section units
    gm_num_collumn  : column offset in section units

    Note: the original bioimaging_subfunctions version called plt.figure() and
    plt.imshow() as a side effect — those are removed here.
    """
    gm_mask_size = gm_size + gm_pad
    gm_pad_size = int((gm_extend_size - gm_size * gm_num_sections - gm_pad) / 2)
    gm_mask_00 = np.zeros((gm_extend_size, gm_extend_size))
    gm_mask_00[
        gm_pad_size:gm_pad_size + gm_mask_size,
        gm_pad_size:gm_pad_size + gm_mask_size,
    ] = 1
    gm_mask = np.roll(
        gm_mask_00,
        (gm_num_row * gm_size, gm_num_collumn * gm_size),
        axis=(0, 1),
    )
    gm_vector = gm_mask.reshape(gm_extend_size ** 2, order='F')
    return np.argwhere(gm_vector).reshape(-1)


# ---------------------------------------------------------------------------
# Butterworth low-pass filter  (vectorised — replaces O(N²) Python loop)
# ---------------------------------------------------------------------------

def butterworth_lowpass_filter(size: int = 100, cutoff: float = 30, order: int = 2) -> np.ndarray:
    """2-D Butterworth low-pass filter.

    Vectorised numpy implementation; produces identical values to the original
    nested-loop version in bioimaging_subfunctions.py.

    Parameters
    ----------
    size   : int   — filter size in pixels (square)
    cutoff : float — cutoff frequency in pixel units (D0)
    order  : int   — filter order n

    Returns
    -------
    float32 ndarray [size, size]
    """
    cx, cy = size // 2, size // 2
    u = np.arange(size) - cx   # row offsets from centre
    v = np.arange(size) - cy   # col offsets from centre
    uu, vv = np.meshgrid(u, v, indexing='ij')
    dist = np.sqrt(uu ** 2 + vv ** 2)
    mask = 1.0 / (1.0 + (dist / cutoff) ** (2 * order))
    return mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Centred mask helper  (used by generate_loose_threshold_masks)
# ---------------------------------------------------------------------------

def _generate_centerized_mask(
    mask_size: int,
    img_size: int,
    option: str = 'butterworth',
    p: float = 10,
) -> np.ndarray:
    """Create a centred mask of *mask_size* embedded in *img_size* × *img_size*."""
    if option == 'squared':
        window = np.ones((mask_size, mask_size), dtype=np.float32)
        return _pad_np(window, img_size)
    elif option == 'butterworth':
        cutoff = int(2 * mask_size / (p / 2))
        window = butterworth_lowpass_filter(size=mask_size, cutoff=cutoff, order=2)
        return _pad_np(window, img_size)
    else:
        raise ValueError(f"Unknown option '{option}'. Use 'squared' or 'butterworth'.")


# ---------------------------------------------------------------------------
# Loose confocal threshold masks
# ---------------------------------------------------------------------------

def generate_loose_threshold_masks(
    lt_mask_size: int,
    lt_img_size: int,
    option: str = 'butterworth',
    p: float = 10,
) -> tuple:
    """Generate the stack of loose confocal threshold masks.

    Parameters
    ----------
    lt_mask_size : int   — ROD window size in pixels
    lt_img_size  : int   — ROI size in pixels
    option       : str   — 'squared' or 'butterworth'
    p            : float — shape parameter for butterworth cutoff

    Returns
    -------
    lt_mask_stack_uncropped : float32 [lt_img_size², lt_img_size+lt_mask_size, lt_img_size+lt_mask_size]
    lt_mask_stack           : float32 [lt_img_size², lt_img_size, lt_img_size]
    """
    lt_uncroped_size = lt_img_size + lt_mask_size
    tmp_A = np.arange(0, lt_img_size, 1)
    tmp_A = tmp_A * np.ones([lt_img_size, 1])
    B_row_idex = tmp_A.reshape(-1, 1, order='C')
    B_collumn_idex = tmp_A.reshape(-1, 1, order='F')
    r_index_data = np.concatenate(
        (B_row_idex + lt_mask_size // 2, B_collumn_idex + lt_mask_size // 2),
        axis=1,
    )
    lt_threshold_mask_0 = _generate_centerized_mask(lt_mask_size, lt_uncroped_size, option, p)

    lt_mask_stack_uncropped = []
    for i in range(len(r_index_data)):
        shifted = np.roll(
            lt_threshold_mask_0,
            shift=(
                int(r_index_data[i, 0] - lt_uncroped_size // 2),
                int(r_index_data[i, 1] - lt_uncroped_size // 2),
            ),
            axis=(0, 1),
        )
        lt_mask_stack_uncropped.append(shifted)

    lt_mask_stack_uncropped = np.array(lt_mask_stack_uncropped)
    lt_mask_stack = lt_mask_stack_uncropped[
        :,
        lt_mask_size // 2:(lt_img_size + lt_mask_size // 2),
        lt_mask_size // 2:(lt_img_size + lt_mask_size // 2),
    ]
    return lt_mask_stack_uncropped, lt_mask_stack


# ---------------------------------------------------------------------------
# Confocal image reconstruction
# ---------------------------------------------------------------------------

def _img_arr2matrix(img_array: np.ndarray) -> np.ndarray:
    """Reshape [N, H, W] image array to [H*W, N] complex matrix (column-major).

    Column i holds the flattened field for illumination position i.
    """
    n, h, w = img_array.shape
    matrix = np.zeros((h * w, n), dtype=np.complex64)
    for i in range(n):
        matrix[:, i] = img_array[i].reshape(h * w, order='F')
    return matrix


def img_arr2confocal_img(img_arr: np.ndarray) -> np.ndarray:
    """Compute confocal image from a [N, H, W] scanning image array.

    Each slice img_arr[i] is the measured field for illumination position i.
    The diagonal of the R-matrix selects the same-position (confocal) response.
    """
    tmp_size = img_arr.shape[1]
    scan_size = int(np.sqrt(img_arr.shape[0]))
    if scan_size != tmp_size:
        tmp_crop = _crop_np(img_arr, scan_size)
    else:
        tmp_crop = img_arr

    rr_matrix = _img_arr2matrix(tmp_crop)
    return np.diag(rr_matrix).reshape((scan_size, scan_size), order='F')


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def next_batch(
    data_x: np.ndarray,
    data_y: np.ndarray,
    batch_size: int,
    previous_batches: set = None,
) -> tuple:
    """Sample a random mini-batch without replacement within an epoch.

    When all indices have been used (*previous_batches* covers the full dataset),
    the epoch resets and a fresh random batch is drawn.

    Parameters
    ----------
    data_x, data_y   : [N, H, W] arrays — input and label data
    batch_size       : number of samples per batch
    previous_batches : set of already-sampled indices (None = start of epoch)

    Returns
    -------
    batch_x, batch_y, batch_indices, updated_previous_batches
    """
    if previous_batches is None:
        previous_batches = set()
    all_indices = set(range(len(data_y)))
    remaining = list(all_indices - previous_batches)
    if len(remaining) < batch_size:
        # Epoch exhausted -- reset and draw from full dataset
        idx = np.random.choice(len(data_y), size=batch_size, replace=False)
    else:
        idx = np.random.choice(remaining, size=batch_size, replace=False)
    previous_batches.update(idx)
    return data_x[idx], data_y[idx], idx, previous_batches


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_text(layers: list, epoch: int, name: str, output_dir: str) -> None:
    """Save layer arrays as .txt checkpoint files (same naming as TF1.x).

    Written as:  {output_dir}/{name}_{i+1}_epoch_{epoch+1}.txt

    np.savetxt formats complex128 as ``(re+imj)`` columns, matching the
    TF1.x checkpoint format so either version can load the other's files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, layer in enumerate(layers):
        path = os.path.join(output_dir, f"{name}_{i + 1}_epoch_{epoch + 1}.txt")
        np.savetxt(path, layer)


def download_loss(loss_data: list, epoch: int, output_dir: str) -> None:
    """Append current loss history to loss.csv and overwrite loss.pdf.

    Parameters
    ----------
    loss_data  : list of scalar loss values (one per epoch completed so far)
    epoch      : current epoch index (0-based), used only for the log message
    output_dir : directory to write loss.csv and loss.pdf
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'loss.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for e, val in enumerate(loss_data):
            writer.writerow([e + 1, float(val)])

    fig, ax = plt.subplots()
    ax.plot(range(1, len(loss_data) + 1), loss_data)
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.savefig(os.path.join(output_dir, 'loss.pdf'))
    plt.close(fig)
    print(f'[epoch {epoch + 1}] Loss saved → {output_dir}')
