"""
multilayer_torch/solver.py
---------------------------
Data-loading, preprocessing, and Adam optimisation loop for the
multi-layer scattering model.

Public classes / functions
--------------------------
    MultiLayerOptimizationTorch   — data container + optimisation loop
    run_from_config(cfg)          — entry point called by cli.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from .loss import (
    coherent_intensity_loss,
    depth_tv_selected,
    entropy_loss,
    l2_loss,
    lateral_tv,
    pearson_loss,
)
from .model import MultiLayerModelTorch
from .physics import generate_layer_params_v2, np_pupil_mask
from .analysis import (
    compute_entropy,
    compute_hf_power,
    compute_pupil_snr,
    compute_sbr,
    save_overfitting_figure,
    save_quality_tracking_figure,
)
from .plotting import (
    save_confocal_figure,
    save_corrected_confocal_figure,
    save_focus_figure,
    save_global_phase_figure,
    save_initial_layers_figure,
    save_intensity_history_figure,
    save_layers_figure,
    save_loss_figure,
    save_pupils_figure,
    save_source_correction_figure,
)
from .utils import (
    center_crop_np,
    change_size_np,
    download_loss,
    generate_mask_indexs,
    generate_loose_threshold_masks,
    img_arr2confocal_img,
    match_ROIandres,
    next_batch,
    save_text,
    _img_arr2matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data pre-loading (used to avoid redundant I/O in scan loops)
# ---------------------------------------------------------------------------

def _load_and_process_data(
    *,
    data_format: str,
    # rrmat mode
    rrmat_path: str = '',
    rrmat_sigma: float = 0,
    rrmat_spot_offset: tuple[int, int] = (0, 0),
    wavelength: float = 0,
    medium_refractive_index: float = 1.0,
    resolution: float = 0,
    # raw-data fallback (used only when rrmat_path does not exist)
    raw_mat_path: str = '',
    raw_N_d: int = 260,
    raw_xc: int = 129,
    raw_yc: int = 104,
    raw_Nx_roi: int | None = None,
    sat_repair: bool = False,
    sat_abs_threshold: int = 25000,
    sat_k_sigma: float = 6.0,
    sat_kernel_size: int = 81,
    sat_n_passes: int = 1,
    # npy mode
    illumination_dir: Path = Path(''),
    illumination_filename: str = '',
    illumination_size: int = 0,
    label_data_dir: Path = Path(''),
    label_filename: str = '',
    labels_size: int = 0,
    # common
    scanning_size: int = 0,
    ROD: int | None = None,
    swap: bool = False,
) -> dict:
    """Load and pre-process experiment data (rrmat or npy).

    Returns a dict containing the fully processed illumination and labels
    arrays, plus any derived parameters (scanning_size, ROD, optical params).
    This function is called once; its output can be reused across multiple
    ``_run_single`` calls (e.g. in a depth-scan loop) to avoid redundant I/O.

    If ``data_format='rrmat'`` and ``rrmat_path`` does not exist, falls back
    to generating one from raw CASS data at ``raw_mat_path`` (see
    ``multilayer_torch.raw2rrmat``).
    """
    if data_format == 'rrmat':
        from .data_loading import load_rrmat, rrmat_to_illum_labels, estimate_rod
        from .raw2rrmat import ensure_rrmat_exists

        # Pre-flight: ensure rrmat_path points to an existing file (or
        # generate it from raw data if raw_mat_path is provided).
        rrmat_path = ensure_rrmat_exists(
            rrmat_path,
            raw_mat_path=raw_mat_path or None,
            raw_N_d=raw_N_d,
            raw_xc=raw_xc,
            raw_yc=raw_yc,
            raw_Nx_roi=raw_Nx_roi,
            sat_repair=sat_repair,
            sat_abs_threshold=sat_abs_threshold,
            sat_k_sigma=sat_k_sigma,
            sat_kernel_size=sat_kernel_size,
            sat_n_passes=sat_n_passes,
        )

        rr, Nx_det, Nx_illum, wl, n0, dx = load_rrmat(
            rrmat_path, wl=wavelength, n0=medium_refractive_index, dx=resolution,
        )
        illum, labels = rrmat_to_illum_labels(
            rr, Nx_det, Nx_illum, sigma=rrmat_sigma, spot_offset=rrmat_spot_offset)
        # Chunked abs-max + in-place divide: avoids a 7 GB abs() temp AND a
        # 14 GB divide-result temp on large rrmats (tight cgroup memory).
        _chunk = 2000
        _abs_max = 0.0
        for _i in range(0, labels.shape[0], _chunk):
            _abs_max = max(_abs_max, float(np.abs(labels[_i:_i + _chunk]).max()))
        labels /= _abs_max

        # Config size warnings
        if illumination_size not in (0, Nx_illum) and illumination_size > 0:
            logger.warning(
                f'Config illumination_size={illumination_size} ignored in rrmat mode '
                f'(using Nx_illum={Nx_illum} from R-matrix)')
        if labels_size not in (0, Nx_det) and labels_size > 0:
            logger.warning(
                f'Config labels_size={labels_size} ignored in rrmat mode '
                f'(using Nx_det={Nx_det} from R-matrix)')

        # Derive scanning_size from R-matrix if not explicitly set
        effective_scanning_size = scanning_size
        if scanning_size == 0 or scanning_size > Nx_illum:
            effective_scanning_size = Nx_illum
            logger.info(f'  scanning_size set to {Nx_illum} (from R-matrix Nx_illum)')

        # Auto-detect ROD
        effective_ROD = ROD
        if ROD is None or ROD == 0:
            effective_ROD = estimate_rod(rr, Nx_det, Nx_illum)

        # Free raw R-matrix now — illum/labels are derived and ROD is known.
        # Saves ~14 GB before the loose_confocal broadcast multiply pushes peak
        # RAM past tight cgroup memory limits.
        del rr
        import gc
        gc.collect()

        # Indexing: select scanning-area subset
        illum_scan = int(np.sqrt(illum.shape[0]))
        idx = generate_mask_indexs(illum_scan, effective_scanning_size)
        illum = illum[idx]
        labels = labels[idx]

        # Swap (transpose R-matrix)
        if swap:
            logger.info('Transposing R-matrix (swap mode)')
            matrix = _img_arr2matrix(labels)
            n, h, w = labels.shape
            mat_t = matrix.T
            arr = np.zeros((n, h, w), dtype=labels.dtype)
            for i in range(n):
                arr[i] = mat_t[i].reshape(h, w, order='F')
            labels = arr

        # Loose confocal gate
        _, loose_masks = generate_loose_threshold_masks(
            effective_ROD, effective_scanning_size, option='butterworth')
        loose_masks = change_size_np(loose_masks, labels.shape[1])
        labels = labels * loose_masks

        return {
            'illumination_data': illum,
            'labels_data': labels,
            'scanning_pixel_size': effective_scanning_size,
            'wavelength': wl,
            'refractive_index': n0,
            'resolution': dx,
        }

    else:
        # Standard .npy mode
        illum = np.load(Path(illumination_dir) / illumination_filename)
        illum = match_ROIandres(illum, resolution, resolution, illumination_size)

        labels = np.load(Path(label_data_dir) / label_filename)
        labels = labels / np.max(np.abs(labels))
        labels = match_ROIandres(labels, resolution, resolution, labels_size)

        effective_scanning_size = scanning_size

        # Indexing — skip when the saved file is already pre-cropped to
        # effective_scanning_size² (otherwise idx, computed against the larger
        # source grid, indexes out of bounds).
        illum_scan = int(np.sqrt(illum.shape[0]))
        idx = generate_mask_indexs(illum_scan, effective_scanning_size)
        n_scan = effective_scanning_size * effective_scanning_size
        if illum.shape[0] == n_scan:
            logger.info(f'  illumination already cropped to {n_scan} sources; skipping')
        else:
            illum = illum[idx]
        if labels.shape[0] == n_scan:
            logger.info(f'  labels already cropped to {n_scan} sources; skipping')
        else:
            labels = labels[idx]

        # Swap
        if swap:
            logger.info('Transposing R-matrix (swap mode)')
            matrix = _img_arr2matrix(labels)
            n, h, w = labels.shape
            mat_t = matrix.T
            arr = np.zeros((n, h, w), dtype=labels.dtype)
            for i in range(n):
                arr[i] = mat_t[i].reshape(h, w, order='F')
            labels = arr

        # Loose confocal gate
        effective_ROD = ROD if ROD is not None else 120
        _, loose_masks = generate_loose_threshold_masks(
            effective_ROD, effective_scanning_size, option='butterworth')
        loose_masks = change_size_np(loose_masks, labels.shape[1])
        labels = labels * loose_masks

        return {
            'illumination_data': illum,
            'labels_data': labels,
            'scanning_pixel_size': effective_scanning_size,
            'wavelength': wavelength,
            'refractive_index': medium_refractive_index,
            'resolution': resolution,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _param_to_complex_np(param: torch.nn.Parameter, ctype: str) -> np.ndarray:
    """Convert a [2, H, W] float32 Parameter to a complex [H, W] ndarray.

    Conversion matches the TF1.x convention so .txt files are cross-compatible.
    """
    arr = param.detach().cpu().numpy()
    if ctype == 'Rectangular':
        return arr[0] + 1j * arr[1]
    elif ctype == 'Euler':
        return arr[0] * np.exp(1j * arr[1])
    elif ctype == 'phase-only complex':
        return np.exp(1j * arr[1])
    else:
        raise ValueError(f"Unknown complex_tensor_type '{ctype}'")




# ---------------------------------------------------------------------------
# MultiLayerOptimizationTorch — data container and optimisation loop
# ---------------------------------------------------------------------------

class MultiLayerOptimizationTorch:
    """Data container and Adam optimisation loop for the PyTorch multi-layer model.

    Workflow
    --------
    1. load_illumination_data / load_labels_data
    2. match_ROIandres_illuminations / match_ROIandres_labels
    3. apply_loose_confocal
    4. indexing_input_data
    5. setup_model_params
    6. initialize_pupil_functions
    7. initialize_object_functions
    8. run_optimization
    """

    def __init__(
        self,
        output_path: Path | str,
        scanning_pixel_size: int,
        layer_positions: np.ndarray,
        wavelength: float,
        refractive_index: float,
        resolution: float,
        NA: float,
        NA_model: float,
        geometry_mode: str = 'Reflection',
        complex_tensor_type: str = 'Rectangular',
    ) -> None:
        self.output_path = Path(output_path)
        self.scanning_pixel_size = scanning_pixel_size
        self.layer_positions = np.asarray(layer_positions, dtype=np.float64)
        self.wavelength = wavelength
        self.refractive_index = refractive_index
        self.resolution = resolution
        self.NA = NA
        self.NA_model = NA_model
        self.num_layers = len(layer_positions)
        self.geometry_mode = geometry_mode
        self.complex_tensor_type = complex_tensor_type

        self.illumination_data: Optional[np.ndarray] = None
        self.labels_data: Optional[np.ndarray] = None
        self.model_params: Optional[dict] = None
        self.initializer: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_illumination_data(self, path: Path | str, filename: str) -> None:
        path = Path(path)
        logger.info(f'Loading illumination: {path / filename}')
        self.illumination_data = np.load(path / filename)

    def load_labels_data(self, path: Path | str, filename: str) -> None:
        path = Path(path)
        logger.info(f'Loading labels: {path / filename}')
        self.labels_data = np.load(path / filename)
        self.labels_data = self.labels_data / np.max(np.abs(self.labels_data))

    def load_from_rrmat(
        self, rrmat_path: str | Path, sigma: float = 0,
        spot_offset: tuple[int, int] = (0, 0),
    ) -> tuple[int, int]:
        """Load data from a single R-matrix .mat file.

        Generates illumination patterns (delta or Gaussian) and labels from
        the reflection matrix. Supports non-square R-matrices where the
        illumination and detection grids differ in size.

        Parameters
        ----------
        rrmat_path : str or Path
            Path to HDF5 .mat file containing the reflection matrix.
        sigma : float
            Gaussian spot width for illumination. 0 = delta function.

        Returns
        -------
        Nx_det : int
            Detection grid dimension (labels size).
        Nx_illum : int
            Illumination grid dimension.
        """
        from .data_loading import load_rrmat, rrmat_to_illum_labels

        rr, Nx_det, Nx_illum, wl, n0, dx = load_rrmat(
            rrmat_path,
            wl=self.wavelength,
            n0=self.refractive_index,
            dx=self.resolution,
        )
        # Override optical params from .mat file if they weren't set in config
        self.wavelength = wl
        self.refractive_index = n0
        self.resolution = dx

        illum, labels = rrmat_to_illum_labels(
            rr, Nx_det, Nx_illum, sigma=sigma, spot_offset=spot_offset)
        self.illumination_data = illum
        # Chunked abs-max + in-place divide: avoids a 7 GB abs() temp AND a
        # 14 GB divide-result temp on large rrmats (tight cgroup memory).
        _chunk = 2000
        _abs_max = 0.0
        for _i in range(0, labels.shape[0], _chunk):
            _abs_max = max(_abs_max, float(np.abs(labels[_i:_i + _chunk]).max()))
        labels /= _abs_max
        self.labels_data = labels

        # Store for potential later use (e.g., ROD estimation)
        self._rr_matrix = rr
        self._rrmat_Nx_det = Nx_det
        self._rrmat_Nx_illum = Nx_illum
        return Nx_det, Nx_illum

    def estimate_rod_from_rrmat(self, threshold_ratio: float = 0.05) -> int:
        """Auto-detect ROD from the loaded R-matrix."""
        from .data_loading import estimate_rod
        if not hasattr(self, '_rr_matrix'):
            raise RuntimeError('No R-matrix loaded. Call load_from_rrmat() first.')
        return estimate_rod(
            self._rr_matrix, self._rrmat_Nx_det, self._rrmat_Nx_illum, threshold_ratio
        )

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def match_ROIandres_illuminations(
        self, current_resolution: float, new_p_size: int
    ) -> None:
        self.illumination_data = match_ROIandres(
            self.illumination_data, current_resolution, self.resolution, new_p_size
        )

    def match_ROIandres_labels(
        self, current_resolution: float, new_p_size: int
    ) -> None:
        self.labels_data = match_ROIandres(
            self.labels_data, current_resolution, self.resolution, new_p_size
        )

    def apply_loose_confocal(self, ROD: int) -> None:
        """Apply a Butterworth loose-confocal gate mask (ROD pixels wide) to labels."""
        _, loose_masks = generate_loose_threshold_masks(
            ROD, self.scanning_pixel_size, option='butterworth'
        )
        new_size = self.labels_data.shape[1]
        logger.info(f'apply_loose_confocal: labels size={new_size}')
        loose_masks = change_size_np(loose_masks, new_size)
        self.labels_data = self.labels_data * loose_masks
        logger.info(f'apply_loose_confocal: done')

    def indexing_input_data(self, swap: bool = False) -> None:
        """Select the scanning-area subset of illumination and labels data.

        Parameters
        ----------
        swap : bool
            If True, transpose the R-matrix (swap input/output channels).
        """
        logger.info('Indexing data to scanning area ...')
        illum_scan = int(np.sqrt(self.illumination_data.shape[0]))
        idx = generate_mask_indexs(illum_scan, self.scanning_pixel_size)
        # Both arrays share the same source indexing (one per illumination point).
        # Either may already be pre-cropped to scanning_pixel_size² in the saved
        # file — detect that case and skip the index op to avoid an OOB error.
        n_scan = self.scanning_pixel_size * self.scanning_pixel_size
        if self.illumination_data.shape[0] == n_scan:
            logger.info(
                f'  illumination already cropped to {n_scan} sources; skipping')
        else:
            self.illumination_data = self.illumination_data[idx]
        if self.labels_data.shape[0] == n_scan:
            logger.info(
                f'  labels already cropped to {n_scan} sources; skipping')
        else:
            self.labels_data = self.labels_data[idx]
        if swap:
            logger.info('Transposing R-matrix (swap mode)')
            matrix = _img_arr2matrix(self.labels_data)
            # Rebuild image array from transposed matrix (column-major reshape)
            n, h, w = self.labels_data.shape
            mat_t = matrix.T      # [n, h*w]
            arr = np.zeros((n, h, w), dtype=self.labels_data.dtype)
            for i in range(n):
                arr[i] = mat_t[i].reshape(h, w, order='F')
            self.labels_data = arr

    # ------------------------------------------------------------------ #
    # Model parameter setup                                                #
    # ------------------------------------------------------------------ #

    def setup_model_params(self) -> dict:
        """Compute propagation geometry for each layer and store in model_params."""
        logger.info('Setting up model parameters')
        illum_px = self.illumination_data.shape[1]
        labels_px = self.labels_data.shape[1]
        FOV = self.resolution * illum_px
        extended_Ls, padded_sizes = generate_layer_params_v2(
            FOV, illum_px, self.layer_positions, self.NA_model, self.refractive_index
        )

        # Crop labels if they exceed the object-plane padded size.
        # Anything outside padded_sizes[-1] can't be predicted by the model.
        max_label_size = int(padded_sizes[-1])
        if labels_px > max_label_size:
            logger.info(
                f'Cropping labels: {labels_px} -> {max_label_size} '
                f'(to fit object-plane padded size)')
            self.labels_data = center_crop_np(self.labels_data, max_label_size)
            labels_px = max_label_size

        pad_size = int((padded_sizes[0] - labels_px) / 2)
        self.model_params = {
            'pad':                       pad_size,
            'illumination_size':         illum_px,
            'size':                      labels_px,
            'wavelength':                self.wavelength,
            'ROI':                       FOV,
            'NA':                        self.NA,
            'propagatedNA':              self.NA_model,
            '# Layers':                  self.num_layers,
            'layer_positions':           self.layer_positions,
            'separate_distances':        self.layer_positions[:-1] - self.layer_positions[1:],
            'medium refractive index':   self.refractive_index,
            'depth':                     float(np.max(self.layer_positions)),
            'layer_sizes':               padded_sizes,
            'layer_lengths':             extended_Ls,
            'scanning_size':             self.scanning_pixel_size,
        }
        return self.model_params

    # ------------------------------------------------------------------ #
    # Initialisation of pupils and object functions                        #
    # ------------------------------------------------------------------ #

    def initialize_pupil_functions(
        self,
        pupil_funcs_type: list[str],
        option: str = 'flat',
        **kwargs,
    ) -> None:
        """Initialise input and output pupil aberration functions.

        Parameters
        ----------
        pupil_funcs_type : ['variable'|'constant', 'variable'|'constant']
        option : 'flat' (zero phase) or 'previous' (load from .txt files)
        path / filename : required when option='previous'
        """
        if self.initializer is None:
            self.initializer = {}
        logger.info(f'Initializing pupils (option={option})')
        sz = int(self.model_params['layer_sizes'][0])

        if option == 'flat':
            # Unit-amplitude, zero-phase pupil stored as [re, im]
            tmp = np.exp(1j * np.zeros((sz, sz), dtype=np.complex64))
            arr = np.array([np.real(tmp), np.imag(tmp)], dtype=np.float32)
            self.initializer['pupil_functions'] = [arr, arr]

        elif option == 'previous':
            path = Path(kwargs['path'])
            filename = kwargs['filename']
            from scipy.ndimage import gaussian_filter
            lmb_eff = self.wavelength / self.refractive_index
            L0 = float(self.model_params['layer_lengths'][0])
            kout_mask = np_pupil_mask(sz, lmb_eff, L0, self.NA)
            kout_mask = gaussian_filter(kout_mask, sigma=1.0)

            import glob
            from .utils import np_fft2, np_ifft2
            # Escape literal brackets in the dir part (e.g. '[1.37, 1.37]')
            # so glob does not treat them as char-class wildcards.
            pattern = str(Path(glob.escape(str(path))) / filename)
            matching = sorted(glob.glob(pattern))
            if not matching:
                raise FileNotFoundError(f'No pupil files matching {pattern}')
            logger.info(f'Found {len(matching)} pupil files')

            pupil_funcs = []
            for fpath in matching:
                tmp = np.loadtxt(fpath, dtype=np.complex64)
                # Resize to current layer size
                if_tmp = np_ifft2(tmp[np.newaxis])[0]
                if_tmp = change_size_np(if_tmp[np.newaxis], sz)[0] if if_tmp.shape[-1] != sz else if_tmp
                tmp_crop = np_fft2(if_tmp[np.newaxis])[0]
                mx = np.max(np.abs(tmp_crop))
                if mx > 0:
                    tmp_crop /= mx
                tmp_masked = np.array(
                    [np.real(tmp_crop) * kout_mask, np.imag(tmp_crop) * kout_mask],
                    dtype=np.float32,
                )
                pupil_funcs.append(tmp_masked)

            if kwargs.get('swap', False):
                pupil_funcs = pupil_funcs[::-1]
            self.initializer['pupil_functions'] = pupil_funcs[:2]

        else:
            raise ValueError("option must be 'flat' or 'previous'")

        self.initializer['pupil_funcs_type'] = pupil_funcs_type
        self.initializer['complex_tensor_type'] = self.complex_tensor_type

    def _gen_confocal_img(self) -> np.ndarray:
        """Generate confocal image from labels and resize to sample layer size.

        Uses offset-based extraction: for each illumination point i at
        (r_ill, c_ill) on the scanning grid, reads the label value at
        (r_ill + offset, c_ill + offset) on the detection grid, where
        offset = (Nx_det - Nx_illum) / 2.
        """
        from .utils import np_fft2, np_ifft2

        Nx_illum = self.scanning_pixel_size
        Nx_det = self.labels_data.shape[1]
        offset = (Nx_det - Nx_illum) // 2
        N_illum = self.labels_data.shape[0]

        confocal = np.zeros((Nx_illum, Nx_illum), dtype=np.complex64)
        for i in range(N_illum):
            r_ill = i % Nx_illum
            c_ill = i // Nx_illum
            r_det = r_ill + offset
            c_det = c_ill + offset
            if r_det < Nx_det and c_det < Nx_det:
                confocal[r_ill, c_ill] = self.labels_data[i, r_det, c_det]

        # Apply k-space filter matching the pupil mask
        sz_sample = int(self.model_params['layer_sizes'][-1])
        lmb_eff = self.wavelength / self.refractive_index
        L_sample = float(self.model_params['layer_lengths'][-1])
        k_mask = np_pupil_mask(confocal.shape[-1], lmb_eff, L_sample, self.NA)
        filtered = np_ifft2((k_mask * np_fft2(confocal[np.newaxis]))[0:1])[0]
        # Resize to sample layer size
        filtered = change_size_np(filtered[np.newaxis], sz_sample)[0]
        mx = np.max(np.abs(filtered))
        return (filtered / mx).astype(np.complex64) if mx > 0 else filtered.astype(np.complex64)

    @staticmethod
    def _layer_to_array(layer: np.ndarray) -> np.ndarray:
        """Convert complex [H, W] to 2-channel [2, H, W] float32 array."""
        return np.array([np.real(layer), np.imag(layer)], dtype=np.float32)

    def initialize_object_functions(
        self,
        input_obj_funcs_type: list[str],
        output_obj_funcs_type: list[str],
        mode: str = 'initial',
        **kwargs,
    ) -> None:
        """Initialise per-layer scattering object functions.

        Parameters
        ----------
        input_obj_funcs_type, output_obj_funcs_type : list of 'variable'|'constant'
        mode : 'initial' (flat+confocal or random) or 'previous' (warm-start)
        init_option : 'flat+confocal' (default) or 'random'
        path / epoch / resolution : required when mode='previous'
        """
        if self.initializer is None:
            self.initializer = {}
        self.initializer['input_obj_funcs_type'] = input_obj_funcs_type
        self.initializer['output_obj_funcs_type'] = output_obj_funcs_type
        self.initializer['complex_tensor_type'] = self.complex_tensor_type

        confocal_img = self._gen_confocal_img()
        self.confocal_img = confocal_img          # stored for initial plot
        input_funcs, output_funcs = [], []
        layer_sizes = self.model_params['layer_sizes']

        # --- mode: initial ---
        if mode == 'initial':
            init_option = kwargs.get('init_option', 'flat+confocal')
            logger.info(f'Initializing object functions (mode=initial, option={init_option})')

            for i in range(self.num_layers - 1):
                sz = int(layer_sizes[i])
                if init_option == 'flat+confocal':
                    flat = np.exp(1j * np.zeros((sz, sz), dtype=np.complex64))
                    input_funcs.append(self._layer_to_array(flat))
                    output_funcs.append(self._layer_to_array(flat))
                elif init_option == 'random':
                    rng = np.random.default_rng()
                    ri = rng.standard_normal((sz, sz)).astype(np.float32)
                    ii = rng.standard_normal((sz, sz)).astype(np.float32)
                    input_funcs.append(np.array([ri, ii]))
                    ro = rng.standard_normal((sz, sz)).astype(np.float32)
                    io = rng.standard_normal((sz, sz)).astype(np.float32)
                    output_funcs.append(np.array([ro, io]))
                else:
                    raise ValueError("init_option must be 'flat+confocal' or 'random'")

            # Sample plane (last layer) — initialised to confocal image
            input_funcs.append(self._layer_to_array(confocal_img))
            output_funcs.append(self._layer_to_array(confocal_img))
            logger.info(f'Initialized {len(output_funcs)} layer pairs')

        # --- mode: previous ---
        elif mode == 'previous':
            path = Path(kwargs['path'])
            epoch = int(kwargs['epoch'])
            res = float(kwargs.get('resolution', self.resolution))
            indices = kwargs.get('indices')
            scanning_layer_order = kwargs.get('scanning_layer_order')
            swap = kwargs.get('swap', False)
            logger.info(f'Loading layers from {path} at epoch {epoch}')

            # Count available layer files.
            # In reflection geometry: N output layers but only N-1 input layers
            # (no input layer at sample plane).
            import glob
            # Escape literal brackets in the dir part so glob's char-class
            # syntax does not mis-interpret e.g. '[1.37, 1.37]' in path names.
            path_glob = Path(glob.escape(str(path)))
            n_saved = len(glob.glob(str(path_glob / f'output_layer_*_epoch_{epoch}.txt')))
            n_in_saved = len(glob.glob(str(path_glob / f'input_layer_*_epoch_{epoch}.txt')))
            if n_saved == 0:
                raise FileNotFoundError(
                    f"No output_layer_*_epoch_{epoch}.txt files in {path}"
                )
            logger.info(
                f'Found {n_saved} output + {n_in_saved} input layer files at epoch {epoch}'
            )

            in_layers_loaded, out_layers_loaded = [], []
            # Load input layers (N-1 in reflection geometry)
            for i in range(n_in_saved):
                fin = path / f'input_layer_{i+1}_epoch_{epoch}.txt'
                in_layers_loaded.append(np.loadtxt(str(fin), dtype=np.complex64))
            # Load output layers (N in reflection geometry)
            for i in range(n_saved):
                fout = path / f'output_layer_{i+1}_epoch_{epoch}.txt'
                out_layers_loaded.append(np.loadtxt(str(fout), dtype=np.complex64))

            if swap:
                in_layers_loaded[:-1], out_layers_loaded[:-1] = (
                    out_layers_loaded[:-1], in_layers_loaded[:-1]
                )

            # Insert extra layers if num_layers > n_saved
            n_diff = self.num_layers - n_saved
            if n_diff > 0:
                if indices is None:
                    idx_arr = np.linspace(self.num_layers - 1, 0, n_diff + 2).astype(int)
                    indices = list(idx_arr[1:-1])
                logger.info(f'Inserting {n_diff} flat layers at indices {indices}')
                for idx in sorted(indices):
                    src_in = in_layers_loaded[idx]
                    src_out = out_layers_loaded[idx]
                    in_layers_loaded.insert(idx, src_in.copy())
                    out_layers_loaded.insert(idx, src_out.copy())

            # Reset scanning layers to flat. Only valid indices touched
            # (no input_layer at sample plane in reflection geometry).
            if scanning_layer_order is not None:
                if not isinstance(scanning_layer_order, (list, tuple)):
                    scanning_layer_order = [scanning_layer_order]
                for idx in scanning_layer_order:
                    sz = int(layer_sizes[idx])
                    flat = np.exp(1j * np.zeros((sz, sz), dtype=np.complex64))
                    if idx < len(in_layers_loaded):
                        in_layers_loaded[idx] = flat
                    if idx < len(out_layers_loaded):
                        out_layers_loaded[idx] = flat

            # Rescale + resize to current layer sizes (process input and output
            # separately since reflection geometry has N output but N-1 input).
            # Encode into the model's OWN parameter representation (the inverse of
            # MultiLayerModelTorch._to_complex): [amp, phase] for Euler /
            # phase-only complex, [re, im] for Rectangular. The old unconditional
            # [re, im] (_layer_to_array) mis-loaded Euler warm-starts — the model
            # reads channel0 as amplitude, channel1 as phase, so a loaded field
            # A·e^{iφ} became A·cosφ·e^{i·A·sinφ} (bead phase sign-flips).
            def _enc(z):
                if self.complex_tensor_type == 'Rectangular':
                    return np.array([np.real(z), np.imag(z)], dtype=np.float32)
                return np.array([np.abs(z), np.angle(z)], dtype=np.float32)
            for i, inp in enumerate(in_layers_loaded):
                if res != self.resolution:
                    inp = match_ROIandres(inp[np.newaxis], res, self.resolution, inp.shape[-1])[0]
                sz = int(layer_sizes[i])
                inp = change_size_np(inp[np.newaxis], sz)[0]
                input_funcs.append(_enc(inp))
            for i, out in enumerate(out_layers_loaded):
                if res != self.resolution:
                    out = match_ROIandres(out[np.newaxis], res, self.resolution, out.shape[-1])[0]
                sz = int(layer_sizes[i])
                out = change_size_np(out[np.newaxis], sz)[0]
                output_funcs.append(_enc(out))

        else:
            raise ValueError("mode must be 'initial' or 'previous'")

        self.initializer['input_object_functions'] = input_funcs
        self.initializer['output_object_functions'] = output_funcs

    # ------------------------------------------------------------------ #
    # Optimisation loop                                                    #
    # ------------------------------------------------------------------ #

    def run_optimization(
        self,
        optimization_hyperparams: dict,
        optimization_epochs: int = 10,
        reg_mode='L2_norm',
        external_model: 'Optional[MultiLayerModelTorch]' = None,
        **kwargs,
    ) -> list[float]:
        """Build the PyTorch model and run the Adam loop.

        Parameters
        ----------
        optimization_hyperparams : dict
            Keys: batch_size, learning_rate, gamma_coeff_1, gamma_coeff_2.
        optimization_epochs : int
        reg_mode : 'L2_norm' | 'L2_norm+TV' | 'L2_norm+layer_correlation' | 'L2_norm+entropy'
        external_model : MultiLayerModelTorch, optional
            If supplied, use this model in place of constructing a fresh one.
            The model is moved to the active device but otherwise left as-is —
            useful for comparison drivers that need to seed multiple cells
            from a shared initial state (`state_dict`-cloned).
        TV_layer_indices : array-like, optional
        use_amp : bool, optional  — enable torch.cuda.amp (default False)
        use_compile : bool, optional — enable torch.compile (default False)

        Returns
        -------
        list of float — per-epoch total loss values
        """
        os.makedirs(self.output_path, exist_ok=True)

        batch_size = optimization_hyperparams['batch_size']
        lr = optimization_hyperparams['learning_rate']
        gamma1 = optimization_hyperparams.get('gamma_coeff_1', 0.0)  # L2_norm
        gamma2 = optimization_hyperparams.get('gamma_coeff_2', 0.0)  # lateral_tv
        gamma3 = optimization_hyperparams.get('gamma_coeff_3', 0.0)  # depth_tv
        gamma4 = optimization_hyperparams.get('gamma_coeff_4', 0.0)  # entropy

        # Normalize reg_mode to a list.
        # Backward compat: old string formats like 'L2_norm+TV' are converted.
        if isinstance(reg_mode, str):
            _compat = {
                'L2_norm': ['L2_norm'],
                'L2_norm+TV': ['L2_norm', 'lateral_tv'],
                'L2_norm+layer_correlation': ['L2_norm', 'lateral_tv'],
                'L2_norm+entropy': ['L2_norm', 'entropy'],
            }
            reg_mode = _compat.get(reg_mode, [reg_mode])
        reg_terms = set(reg_mode)

        use_amp = kwargs.get('use_amp', False)
        use_compile = kwargs.get('use_compile', False)
        TV_layer_indices = list(kwargs.get('TV_layer_indices', []))
        use_per_source = kwargs.get('use_per_source_correction', False)
        illumination_mode = kwargs.get('illumination_mode', 'single_focus')
        speckle_cfg = kwargs.get('speckle', {})
        save_every = kwargs.get('save_every', 1)  # save checkpoints/figures every N epochs
        # At each checkpoint, optionally apply a complex median filter to all
        # scattering layers (not the sample). 0 = disabled (default).
        phase_med_filter_k = int(kwargs.get('phase_median_filter_kernel', 0))
        # Optionally restrict the filter to specific epochs (1-indexed).
        # None / unset → fire at every checkpoint where (epoch+1) % save_every == 0.
        # E.g. [5] → apply once at epoch 5; [5, 10] → at epochs 5 and 10.
        phase_med_filter_at = kwargs.get('phase_median_filter_at_epochs', None)
        if phase_med_filter_at is not None:
            phase_med_filter_at = set(int(e) for e in phase_med_filter_at)

        # Model B (coherent intensity loss) config — only active for speckle modes.
        intensity_cfg = kwargs.get('intensity_loss', {}) or {}
        intensity_enabled = bool(intensity_cfg.get('enabled', False))
        intensity_weight = float(intensity_cfg.get('weight', 1.0))
        intensity_eps_alpha = float(intensity_cfg.get('wiener_eps_alpha', 0.02))
        # Optional weight schedule. Two formats are supported:
        #
        # 1) Multi-stage list (most general):
        #    intensity_loss:
        #      schedule:
        #        - {epoch: 0, weight: 0.1}
        #        - {epoch: 3, weight: 1.0}
        #        - {epoch: 5, weight: 10.0}
        #    The weight at training epoch e is the weight of the last
        #    schedule entry whose `epoch` is ≤ e.
        #
        # 2) Two-stage warmup (backward compat with earlier YAMLs):
        #      epochs [0, warmup_epochs)  : warmup_weight
        #      epochs [warmup_epochs, ∞)  : weight
        #
        # Defaults (no schedule, warmup_epochs=0) ⇒ no schedule, weight used throughout.
        intensity_schedule = intensity_cfg.get('schedule', None)
        if intensity_schedule:
            # Normalise + sort by epoch
            intensity_schedule = sorted(
                [(int(s['epoch']), float(s['weight'])) for s in intensity_schedule],
                key=lambda x: x[0],
            )
        intensity_warmup_weight = float(intensity_cfg.get('warmup_weight', intensity_weight))
        intensity_warmup_epochs = int(intensity_cfg.get('warmup_epochs', 0))

        def _intensity_weight_at_epoch(e: int) -> float:
            """Pick the user-facing weight for training epoch `e` according
            to the configured schedule (or warmup, or constant)."""
            if intensity_schedule:
                w = intensity_weight  # if no stage matches yet, use default
                for ep_threshold, w_at in intensity_schedule:
                    if e >= ep_threshold:
                        w = w_at
                    else:
                        break
                return w
            if intensity_warmup_epochs > 0:
                return intensity_warmup_weight if e < intensity_warmup_epochs else intensity_weight
            return intensity_weight

        # Layer sizes
        layer_sizes = self.model_params['layer_sizes']
        input_sizes = list(layer_sizes[: self.num_layers - 1].astype(int))
        output_sizes = list(layer_sizes.astype(int))
        pupil_size = int(layer_sizes[0])

        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Optimizing on {device}  amp={use_amp}  compile={use_compile}')

        # Build model (or accept a caller-supplied one for shared-init comparisons)
        if external_model is not None:
            model = external_model.to(device)
        else:
            model = MultiLayerModelTorch(self.model_params, self.initializer).to(device)
        if use_compile:
            model = torch.compile(model)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device == 'cuda')

        # Speckle / partial_speckle / angular mode setup
        if illumination_mode in ('speckle', 'partial_speckle', 'angular'):
            N_src = len(self.illumination_data)
            Nx_illum = self.illumination_data.shape[1]   # illumination spatial size
            Nx_det = self.labels_data.shape[1]            # detection spatial size
            speckle_steps = speckle_cfg.get('steps_per_epoch', 100)
            speckle_batch = speckle_cfg.get('proj_batch', 50)
            grad_clip_norm = speckle_cfg.get('grad_clip_max_norm', 1.0)
            gpu_data = speckle_cfg.get('gpu_data', True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=optimization_epochs, eta_min=lr * 0.01)

            # Partial speckle params (only used when illumination_mode == 'partial_speckle')
            partial_p = speckle_cfg.get('partial_speckle_p', 0.5)
            partial_phase = speckle_cfg.get('partial_speckle_phase', 'random')

            # Normalization: sqrt(N_src) for full speckle / angular,
            # sqrt(p * N_src) for partial. Per the angular-mode design,
            # |v_s|² sums to N_src for unit-modulus phase ramps too, so
            # the same /√N_src normalization keeps numerical scale
            # consistent across modes (see ToolSearch design notes).
            if illumination_mode == 'partial_speckle':
                speckle_norm = np.sqrt(partial_p * N_src)
            else:
                speckle_norm = np.sqrt(N_src)

            # Flatten and move to device for matmul projection
            _dev = device if gpu_data else 'cpu'
            illum_flat = torch.tensor(
                self.illumination_data.reshape(N_src, -1),
                dtype=torch.complex64, device=_dev)
            labels_flat = torch.tensor(
                self.labels_data.reshape(N_src, -1),
                dtype=torch.complex64, device=_dev)

            # ---- Angular mode: precompute NA-allowed plane-wave angles ----
            # Sources sit on a `scanning_size × scanning_size` grid (row-major:
            # source index s = i * scanning_size + j). Each phase-ramp v_s =
            # exp(i·2π·(Kx·i + Ky·j) / scanning_size) over the source grid
            # synthesises a tilted plane-wave illumination. The pair (Kx, Ky)
            # is one pixel in the source array's Fourier dual; physical NA
            # cuts this dual to a disk of radius NA · L_scan / lmb_eff in
            # pixel units, where L_scan = scanning_size · self.resolution is
            # the source-grid physical extent (sources sampled at the
            # illumination pixel resolution).
            angular_n_grid = None
            allowed_kxy_t = None
            src_i = None
            src_j = None
            if illumination_mode == 'angular':
                angular_n_grid = int(round(np.sqrt(N_src)))
                if angular_n_grid * angular_n_grid != N_src:
                    raise ValueError(
                        f'angular mode requires N_src=scanning_size² '
                        f'(got N_src={N_src}, not a perfect square)')
                lmb_eff = self.wavelength / self.refractive_index
                L_scan_um = angular_n_grid * self.resolution
                kmask_radius_px = self.NA * L_scan_um / lmb_eff
                _kxs = np.arange(-angular_n_grid // 2, angular_n_grid // 2)
                _kyy, _kxx = np.meshgrid(_kxs, _kxs, indexing='ij')
                _na_disk = (_kxx ** 2 + _kyy ** 2) < kmask_radius_px ** 2
                _allowed_kxy = np.argwhere(_na_disk) - angular_n_grid // 2
                allowed_kxy_t = torch.from_numpy(_allowed_kxy).to(
                    device=device, dtype=torch.float32)
                _ii, _jj = np.meshgrid(
                    np.arange(angular_n_grid),
                    np.arange(angular_n_grid),
                    indexing='ij')
                src_i = torch.from_numpy(_ii.flatten()).to(
                    device=device, dtype=torch.float32)
                src_j = torch.from_numpy(_jj.flatten()).to(
                    device=device, dtype=torch.float32)
                logger.info(
                    f'Angular mode: scanning_size={angular_n_grid}, '
                    f'NA-allowed angles={_allowed_kxy.shape[0]}/{N_src} '
                    f'(disk radius={kmask_radius_px:.2f}px), '
                    f'{speckle_steps} steps/epoch, '
                    f'{speckle_batch} projections/step, '
                    f'Nx_illum={Nx_illum}, Nx_det={Nx_det}, '
                    f'gpu_data={gpu_data}, grad_clip={grad_clip_norm}')
            elif illumination_mode == 'partial_speckle':
                logger.info(
                    f'Partial speckle mode: p={partial_p}, phase={partial_phase}, '
                    f'{speckle_steps} steps/epoch, '
                    f'{speckle_batch} projections/step, '
                    f'Nx_illum={Nx_illum}, Nx_det={Nx_det}, '
                    f'gpu_data={gpu_data}, grad_clip={grad_clip_norm}')
            else:
                logger.info(
                    f'Speckle mode: {speckle_steps} steps/epoch, '
                    f'{speckle_batch} projections/step, '
                    f'Nx_illum={Nx_illum}, Nx_det={Nx_det}, '
                    f'gpu_data={gpu_data}, grad_clip={grad_clip_norm}')

        total_batch = max(1, len(self.labels_data) // batch_size)
        loss_history: list[float] = []
        pearson_history: list[float] = []   # Pearson component
        reg_sum_history: list[float] = []   # total weighted regularization
        # Per-term regularization histories (each already multiplied by gamma)
        reg_l2_history: list[float] = []
        reg_ltv_history: list[float] = []
        reg_dtv_history: list[float] = []
        reg_ent_history: list[float] = []
        intensity_history: list[float] = []  # Model B coherent intensity loss
        hf_history: list[float] = []        # HF power fraction of sample layer
        sbr_history: list[float] = []       # SBR of sample layer
        entropy_history: list[float] = []   # Shannon entropy of sample layer intensity
        pupil_snr_in_history:  list[float] = []   # amplitude SNR of input pupil
        pupil_snr_out_history: list[float] = []   # amplitude SNR of output pupil

        # --- Initial plots (once, before optimization starts) ---
        self._save_initial_plots(model)

        # --- Model B auto-balance: compute a scale factor at step 0 so that
        # user's weight=1.0 means "Model B contribution matches Pearson
        # magnitude". The schedule (warmup_weight for first warmup_epochs
        # epochs, then weight) multiplies this scale per-epoch — see the
        # `intensity_effective_weight = ...` update inside the epoch loop.
        intensity_scale = 1.0  # passthrough when disabled
        intensity_effective_weight = intensity_weight
        if intensity_enabled and illumination_mode in ('speckle', 'partial_speckle', 'angular'):
            with torch.no_grad():
                _rp = torch.rand(speckle_batch, N_src, device=device) * (2 * np.pi)
                _v = torch.exp(1j * _rp).to(torch.complex64)
                if use_per_source and hasattr(model, 'source_phase'):
                    _sc = torch.exp(model.source_log_amp + 1j * model.source_phase).to(torch.complex64)
                    _w = _v * _sc[None, :]
                else:
                    _w = _v
                _il = illum_flat.to(device) if not gpu_data else illum_flat
                _lb = labels_flat.to(device) if not gpu_data else labels_flat
                _x = (_w @ _il).reshape(speckle_batch, Nx_illum, Nx_illum) / speckle_norm
                _y = (_v @ _lb).reshape(speckle_batch, Nx_det, Nx_det) / speckle_norm

                _out, _bw = model(_x, _y)
                _pearson = pearson_loss(_bw, _out)
                _ein, _eback = model.forward_sample_plane_fields(_x, _y)
                _crop = min(int(self.scanning_pixel_size), int(model.padded_sizes[-1]))
                _cost_B = coherent_intensity_loss(
                    _ein, _eback, crop_size=_crop, eps_alpha=intensity_eps_alpha,
                )

                intensity_scale = abs(_pearson.item()) / max(abs(_cost_B.item()), 1e-30)
                # Initial effective weight uses the schedule lookup at epoch 0.
                _initial_w = _intensity_weight_at_epoch(0)
                intensity_effective_weight = _initial_w * intensity_scale
                if intensity_schedule:
                    schedule_str = (
                        'schedule=' + ','.join(f'(e≥{e}:w={w})' for e, w in intensity_schedule)
                    )
                elif intensity_warmup_epochs > 0:
                    schedule_str = (
                        f'warmup: weight={intensity_warmup_weight} for {intensity_warmup_epochs} ep, '
                        f'then weight={intensity_weight}'
                    )
                else:
                    schedule_str = f'constant weight={intensity_weight}'
                logger.info(
                    f'Model B auto-balance: |pearson|={abs(_pearson.item()):.4e}, '
                    f'|cost_B|={abs(_cost_B.item()):.4e}, scale={intensity_scale:.4e}, '
                    f'initial effective weight={intensity_effective_weight:.4e} '
                    f'({schedule_str})'
                )

        for epoch in tqdm(range(optimization_epochs), desc='Epochs'):
            # Multi-stage weight schedule for Model B (see _intensity_weight_at_epoch).
            if intensity_enabled and illumination_mode in ('speckle', 'partial_speckle', 'angular'):
                _w_now = _intensity_weight_at_epoch(epoch)
                new_eff = _w_now * intensity_scale
                if abs(new_eff - intensity_effective_weight) > 1e-30:
                    logger.info(
                        f'Model B schedule (epoch {epoch+1}): user_weight={_w_now} '
                        f'→ effective weight={new_eff:.4e}'
                    )
                intensity_effective_weight = new_eff

            previous_batches: set = set()
            epoch_loss = 0.0
            epoch_pearson = 0.0
            epoch_reg_l2 = 0.0
            epoch_reg_ltv = 0.0
            epoch_reg_dtv = 0.0
            epoch_reg_ent = 0.0
            epoch_intensity = 0.0

            # Determine number of steps and batch size for this epoch
            if illumination_mode in ('speckle', 'partial_speckle', 'angular'):
                n_steps = speckle_steps
                effective_batch = speckle_batch
            else:
                n_steps = total_batch
                effective_batch = batch_size

            for step in range(n_steps):

                # ---- Batch generation (mode-dependent) ----
                if illumination_mode == 'speckle':
                    # Full speckle: random phase-only projections (|v|=1)
                    random_phase = torch.rand(
                        speckle_batch, N_src, device=device) * (2 * np.pi)
                    v = torch.exp(1j * random_phase).to(torch.complex64)

                    # Per-source correction applied at projection level
                    if use_per_source and hasattr(model, 'source_phase'):
                        source_complex = torch.exp(
                            model.source_log_amp + 1j * model.source_phase
                        ).to(torch.complex64)
                        w = v * source_complex[None, :]
                    else:
                        w = v

                    il = illum_flat.to(device) if not gpu_data else illum_flat
                    lb = labels_flat.to(device) if not gpu_data else labels_flat
                    x = (w @ il).reshape(speckle_batch, Nx_illum, Nx_illum) / speckle_norm
                    y = (v @ lb).reshape(speckle_batch, Nx_det, Nx_det) / speckle_norm
                    src_idx = None

                elif illumination_mode == 'partial_speckle':
                    # Partial speckle: binary amplitude mask (0 or 1),
                    # each source selected with probability partial_p.
                    mask = (torch.rand(speckle_batch, N_src, device=device) < partial_p).to(torch.float32)

                    # Phase: either zero (constant phasor) or random
                    if partial_phase == 'random':
                        phase = torch.rand(
                            speckle_batch, N_src, device=device) * (2 * np.pi)
                        v = (mask * torch.exp(1j * phase)).to(torch.complex64)
                    else:
                        # zero phase: v = mask (real-valued, cast to complex)
                        v = mask.to(torch.complex64)

                    # Per-source correction applied at projection level
                    if use_per_source and hasattr(model, 'source_phase'):
                        source_complex = torch.exp(
                            model.source_log_amp + 1j * model.source_phase
                        ).to(torch.complex64)
                        w = v * source_complex[None, :]
                    else:
                        w = v

                    il = illum_flat.to(device) if not gpu_data else illum_flat
                    lb = labels_flat.to(device) if not gpu_data else labels_flat
                    x = (w @ il).reshape(speckle_batch, Nx_illum, Nx_illum) / speckle_norm
                    y = (v @ lb).reshape(speckle_batch, Nx_det, Nx_det) / speckle_norm
                    src_idx = None

                elif illumination_mode == 'angular':
                    # Angular: pick `speckle_batch` random (Kx, Ky) inside the
                    # NA disk; each gives a phase ramp over the source grid
                    # that synthesises a tilted plane-wave illumination.
                    n_allowed = allowed_kxy_t.shape[0]
                    sel = torch.randint(0, n_allowed, (speckle_batch,), device=device)
                    Kx = allowed_kxy_t[sel, 0]  # [B], centred integer angles
                    Ky = allowed_kxy_t[sel, 1]
                    # phase[b, s] = 2π/N · (Kx[b]·i_s + Ky[b]·j_s)
                    phase = (2.0 * np.pi / angular_n_grid) * (
                        Kx[:, None] * src_i[None, :]
                        + Ky[:, None] * src_j[None, :]
                    )
                    v = torch.exp(1j * phase).to(torch.complex64)

                    # Per-source correction applied at projection level
                    if use_per_source and hasattr(model, 'source_phase'):
                        source_complex = torch.exp(
                            model.source_log_amp + 1j * model.source_phase
                        ).to(torch.complex64)
                        w = v * source_complex[None, :]
                    else:
                        w = v

                    il = illum_flat.to(device) if not gpu_data else illum_flat
                    lb = labels_flat.to(device) if not gpu_data else labels_flat
                    x = (w @ il).reshape(speckle_batch, Nx_illum, Nx_illum) / speckle_norm
                    y = (v @ lb).reshape(speckle_batch, Nx_det, Nx_det) / speckle_norm
                    src_idx = None
                else:
                    # Single-focus: individual source/detector pairs
                    batch_x, batch_y, batch_idx, previous_batches = next_batch(
                        self.illumination_data, self.labels_data,
                        batch_size, previous_batches,
                    )
                    x = torch.from_numpy(batch_x).to(device)
                    y = torch.from_numpy(batch_y).to(device)
                    src_idx = torch.from_numpy(batch_idx).to(device) if use_per_source else None

                # ---- Forward + loss (shared) ----
                optimizer.zero_grad()
                ctx = torch.amp.autocast('cuda', enabled=use_amp and device == 'cuda')
                with ctx:
                    output_imgs, bw_gt = model(x, y, source_idx=src_idx)

                    # Pearson loss (tracked separately for plotting)
                    pearson_val = pearson_loss(bw_gt, output_imgs)
                    loss = pearson_val
                    step_l2 = 0.0
                    step_ltv = 0.0
                    step_dtv = 0.0
                    step_ent = 0.0

                    # Collect learnable complex layers for regularisation
                    named = model.named_complex_layers()
                    in_complex = [c for n, c, _ in named if 'input' in n]
                    out_complex = [c for n, c, _ in named if 'output' in n]
                    in_sz = [s for n, _, s in named if 'input' in n]
                    out_sz = [s for n, _, s in named if 'output' in n]

                    # --- L2 regularisation (gamma1) ---
                    # Pupils are now Euler-parameterised (A·exp(iφ)); under that
                    # parameterisation, |pupil|² = A² regularises only amplitude
                    # while leaving phase unconstrained. That's not the original
                    # intent of this term, so pupils are dropped from L2 here.
                    # L2 still applies to scattering / sample layers, which use
                    # the Rectangular parameterisation where |x|² = re² + im²
                    # is well-defined.
                    if 'L2_norm' in reg_terms and (in_complex or out_complex):
                        l2_reg = gamma1 * (
                            l2_loss(in_complex, in_sz)
                            + l2_loss(out_complex, out_sz)
                        )
                        loss = loss + l2_reg
                        step_l2 = l2_reg.item()

                    # --- Lateral TV regularisation (gamma2) ---
                    if 'lateral_tv' in reg_terms and TV_layer_indices and out_complex:
                        tv_c = [out_complex[i] for i in TV_layer_indices if i < len(out_complex)]
                        tv_s = [out_sz[i] for i in TV_layer_indices if i < len(out_sz)]
                        if tv_c:
                            ltv_term = gamma2 * lateral_tv(tv_c, tv_s)
                            loss = loss + ltv_term
                            step_ltv = ltv_term.item()

                    # --- Depth TV regularisation (gamma3) ---
                    if 'depth_tv' in reg_terms and TV_layer_indices and out_complex:
                        dtv_term = gamma3 * depth_tv_selected(
                            out_complex, out_sz, TV_layer_indices)
                        loss = loss + dtv_term
                        step_dtv = dtv_term.item()

                    # --- Entropy regularisation on sample layer (gamma4) ---
                    if 'entropy' in reg_terms and out_complex:
                        ent_term = gamma4 * entropy_loss(out_complex[-1])
                        loss = loss + ent_term
                        step_ent = ent_term.item()

                    # --- Model B: coherent intensity loss (speckle modes only) ---
                    # Active only when intensity_loss.enabled and we have a
                    # batch of speckle realisations to coherently sum over.
                    step_intensity = 0.0
                    if intensity_enabled and illumination_mode in ('speckle', 'partial_speckle', 'angular'):
                        # Model B uses ONLY the phase part of per-source
                        # correction. Amplitude factor exp(source_log_amp) is
                        # excluded from Model B's data_x; Pearson still
                        # updates source_log_amp via Model A's `x`.
                        if use_per_source and hasattr(model, 'source_phase'):
                            sc_B = torch.exp(1j * model.source_phase).to(torch.complex64)
                            w_B = v * sc_B[None, :]
                            x_B = (
                                (w_B @ il).reshape(speckle_batch, Nx_illum, Nx_illum)
                                / speckle_norm
                            )
                        else:
                            x_B = x

                        E_in_s, E_back_s = model.forward_sample_plane_fields(
                            x_B, y, source_idx=src_idx,
                        )
                        # Crop to scanning_size (actual scan area), bounded
                        # by the sample-plane padded size.
                        crop = min(int(self.scanning_pixel_size), int(model.padded_sizes[-1]))
                        cost_B = coherent_intensity_loss(
                            E_in_s, E_back_s,
                            crop_size=crop,
                            eps_alpha=intensity_eps_alpha,
                        )
                        loss = loss + intensity_effective_weight * cost_B
                        step_intensity = cost_B.item()

                # ---- Backward + step (shared) ----
                scaler.scale(loss).backward()
                # Mask pupil gradients outside NA aperture
                if hasattr(model, 'mask_pupil_gradients'):
                    model.mask_pupil_gradients()
                if illumination_mode in ('speckle', 'partial_speckle', 'angular'):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                epoch_pearson += pearson_val.item()
                epoch_reg_l2 += step_l2
                epoch_reg_ltv += step_ltv
                epoch_reg_dtv += step_dtv
                epoch_reg_ent += step_ent
                epoch_intensity += step_intensity

            # LR scheduler step (speckle mode only)
            if illumination_mode in ('speckle', 'partial_speckle', 'angular'):
                scheduler.step()

            avg_loss = epoch_loss / n_steps
            loss_history.append(avg_loss)
            pearson_history.append(abs(epoch_pearson / (effective_batch * n_steps)))

            # Per-term regularization (averaged over steps)
            avg_l2 = epoch_reg_l2 / n_steps
            avg_ltv = epoch_reg_ltv / n_steps
            avg_dtv = epoch_reg_dtv / n_steps
            avg_ent = epoch_reg_ent / n_steps
            avg_intensity = epoch_intensity / n_steps
            reg_l2_history.append(avg_l2)
            reg_ltv_history.append(avg_ltv)
            reg_dtv_history.append(avg_dtv)
            reg_ent_history.append(avg_ent)
            intensity_history.append(avg_intensity)
            reg_sum_history.append(avg_l2 + avg_ltv + avg_dtv + avg_ent)

            logger.info(
                f'Epoch {epoch+1}/{optimization_epochs}  loss={avg_loss:.6f}'
                f'  pearson={pearson_history[-1]:.6f}'
                f'  reg_sum={reg_sum_history[-1]:.6f}'
                + (f'  cost_B={intensity_history[-1]:.4e}' if intensity_enabled else '')
            )

            # --- Quality metrics (computed every epoch for complete histories) ---
            self._compute_quality_metrics(
                model, epoch,
                hf_history, sbr_history, entropy_history,
                pupil_snr_in_history, pupil_snr_out_history,
            )

            # --- Save checkpoints, figures, loss CSV every save_every epochs ---
            # Always save on the last epoch regardless of save_every.
            if (epoch + 1) % save_every == 0 or (epoch + 1) == optimization_epochs:
                # Optional: smooth scattering layers via complex median filter.
                # By default fires at every checkpoint; if `phase_median_filter
                # _at_epochs` is set, only fires at those (1-indexed) epochs.
                if phase_med_filter_k > 0 and (
                    phase_med_filter_at is None
                    or (epoch + 1) in phase_med_filter_at
                ):
                    self._median_filter_scattering_phase(model, phase_med_filter_k)
                self._save_checkpoints(model, epoch)
                download_loss(loss_history, epoch, str(self.output_path))
                self._save_epoch_plots(
                    model, epoch,
                    pearson_history, reg_sum_history,
                    reg_l2_history, reg_ltv_history,
                    reg_dtv_history, reg_ent_history,
                    hf_history, sbr_history, entropy_history,
                    pupil_snr_in_history, pupil_snr_out_history,
                )

                # Model B: save corrected-confocal image and intensity tracking
                if intensity_enabled and illumination_mode in ('speckle', 'partial_speckle', 'angular'):
                    try:
                        with torch.no_grad():
                            _Ein, _Eback = model.forward_sample_plane_fields(
                                x, y, source_idx=src_idx,
                            )
                            _abs_sq = _Ein.abs() ** 2
                            _eps_k = intensity_eps_alpha * _abs_sq.mean(dim=(-2, -1), keepdim=True)
                            _I_k = _Eback * _Ein.conj() / (_abs_sq + _eps_k)
                            _S = _I_k.sum(dim=0)
                            _crop = min(int(self.scanning_pixel_size), int(model.padded_sizes[-1]))
                            _h, _w = _S.shape[-2], _S.shape[-1]
                            _r0 = _h // 2 - _crop // 2
                            _c0 = _w // 2 - _crop // 2
                            _S_np = _S[_r0:_r0 + _crop, _c0:_c0 + _crop].detach().cpu().numpy()
                        sample_res = (
                            float(self.model_params['layer_lengths'][-1])
                            / float(self.model_params['layer_sizes'][-1])
                        )
                        save_corrected_confocal_figure(
                            _S_np, sample_res,
                            self.output_path / f'corrected_confocal_epoch{epoch+1:02d}.png',
                            epoch=epoch,
                        )
                    except Exception as exc:
                        logger.warning('corrected confocal figure failed: %s', exc)

                    try:
                        save_intensity_history_figure(
                            intensity_history,
                            self.output_path / 'intensity_history.png',
                        )
                    except Exception as exc:
                        logger.warning('intensity history figure failed: %s', exc)

            if (epoch + 1) % 5 == 0:
                plt.close('all')

        return loss_history

    def _save_initial_plots(self, model: MultiLayerModelTorch) -> None:
        """Save initial_layers.png and confocal_image.png before optimization."""
        try:
            in_funcs = self.initializer['input_object_functions']
            out_funcs = self.initializer['output_object_functions']
            save_initial_layers_figure(
                in_funcs, out_funcs,
                self.complex_tensor_type,
                self.output_path,
            )
        except Exception as exc:
            logger.warning('Could not save initial layers figure: %s', exc)

        try:
            if hasattr(self, 'confocal_img') and self.confocal_img is not None:
                layer_lengths = self.model_params['layer_lengths']
                layer_sizes   = self.model_params['layer_sizes']
                sample_res = float(layer_lengths[-1]) / float(layer_sizes[-1])
                confocal_o = change_size_np(self.confocal_img,self.scanning_pixel_size)
                save_confocal_figure(
                    confocal_o,
                    sample_res,
                    self.output_path / 'confocal_image.png',
                    scanning_size=self.scanning_pixel_size,
                )
        except Exception as exc:
            logger.warning('Could not save confocal figure: %s', exc)

    def _save_epoch_plots(
        self,
        model: MultiLayerModelTorch,
        epoch: int,
        pearson_history: list[float],
        reg_sum_history: list[float],
        reg_l2_history: list[float],
        reg_ltv_history: list[float],
        reg_dtv_history: list[float],
        reg_ent_history: list[float],
        hf_history: list[float],
        sbr_history: list[float],
        entropy_history: list[float],
        pupil_snr_in_history: list[float],
        pupil_snr_out_history: list[float],
    ) -> None:
        """Save all per-epoch figures (layers, focus, pupils, global phase, loss, quality)."""
        layer_lengths = self.model_params['layer_lengths']
        layer_sizes   = self.model_params['layer_sizes']
        layer_pos     = self.layer_positions

        # Per-layer pixel size (µm/pixel) for scale bars
        per_layer_res = [
            float(layer_lengths[i]) / float(layer_sizes[i])
            for i in range(self.num_layers)
        ]

        in_np, out_np = self._layers_to_numpy(model)
        # in_np  : N-1 complex arrays (scattering, no sample)
        # out_np : N   complex arrays (including sample at index -1)

        scatt_res  = per_layer_res[:-1]           # N-1 resolutions for scattering layers
        scatt_pos  = list(layer_pos[:-1])         # N-1 depths
        sample_res = per_layer_res[-1]

        # --- Input scattering layers ---
        try:
            save_layers_figure(
                in_np,
                scatt_pos,
                scatt_res,
                self.output_path / f'input_layers_epoch{epoch + 1:02d}.png',
                title='Input Layers',
                epoch=epoch,
            )
        except Exception as exc:
            logger.warning('input layers figure failed: %s', exc)

        # --- Output scattering layers (exclude sample plane) ---
        try:
            save_layers_figure(
                out_np[:-1],
                scatt_pos,
                scatt_res,
                self.output_path / f'output_layers_epoch{epoch + 1:02d}.png',
                title='Output Layers',
                epoch=epoch,
            )
        except Exception as exc:
            logger.warning('output layers figure failed: %s', exc)

        # --- Focus-plane object (sample layer at z = 0) ---
        try:
            sample = out_np[-1]
            input_sample = in_np[-1]
            save_focus_figure(
                input_sample,sample,
                sample_res,
                self.output_path / f'focus_object_epoch{epoch + 1:02d}.png',
                epoch=epoch,
            )
        except Exception as exc:
            logger.warning('focus figure failed: %s', exc)

        # --- Pupil aberrations (masked by kin_mask for clean visualization) ---
        try:
            pupil_res = float(layer_lengths[0]) / float(layer_sizes[0])
            kin_mask_np = model.kin_mask.detach().cpu().numpy()
            p_in_np = model.pupil_in.detach().cpu().numpy()
            p_out_np = model.pupil_out.detach().cpu().numpy()
            p_in_np = p_in_np * kin_mask_np
            p_out_np = p_out_np * kin_mask_np
            save_pupils_figure(
                p_in_np, p_out_np,
                pupil_res,
                self.output_path / f'pupils_epoch{epoch + 1:02d}.png',
                epoch=epoch,
            )
        except Exception as exc:
            logger.warning('pupils figure failed: %s', exc)

        # --- Global phase map (skip when frozen — e.g. per-source correction active) ---
        if model.global_phase_map.requires_grad:
            try:
                gpm = np.angle(np.exp(1j * model.global_phase_map.detach().cpu().numpy()))
                save_global_phase_figure(gpm, epoch, self.output_path)
            except Exception as exc:
                logger.warning('global phase figure failed: %s', exc)

        # --- Per-source correction maps ---
        if hasattr(model, 'use_per_source_correction') and model.use_per_source_correction:
            try:
                sp = model.source_phase.detach().cpu().numpy()
                sa = model.source_log_amp.detach().cpu().numpy()
                save_source_correction_figure(
                    sp, sa, self.scanning_pixel_size, epoch, self.output_path)
            except Exception as exc:
                logger.warning('source correction figure failed: %s', exc)

        # --- Loss curve (twinx: Pearson / reg_sum) ---
        try:
            save_loss_figure(
                pearson_history, reg_sum_history,
                self.output_path,
            )
        except Exception as exc:
            logger.warning('loss figure failed: %s', exc)

        # --- Regularization breakdown figure ---
        try:
            reg_dict = {}
            if any(v != 0 for v in reg_l2_history):
                reg_dict['L2'] = reg_l2_history
            if any(v != 0 for v in reg_ltv_history):
                reg_dict['lateral_tv'] = reg_ltv_history
            if any(v != 0 for v in reg_dtv_history):
                reg_dict['depth_tv'] = reg_dtv_history
            if any(v != 0 for v in reg_ent_history):
                reg_dict['entropy'] = reg_ent_history
            if reg_dict:
                self._save_reg_breakdown_figure(reg_dict, epoch)
        except Exception as exc:
            logger.warning('regularization breakdown figure failed: %s', exc)

        # --- Quality metrics figures (uses histories populated by _compute_quality_metrics) ---
        try:
            epoch_labels = list(range(1, len(hf_history) + 1))
            save_quality_tracking_figure(epoch_labels, hf_history, sbr_history,
                                         self.output_path,
                                         entropy_history=entropy_history)
            save_overfitting_figure(epoch_labels, hf_history,
                                    pupil_snr_in_history, pupil_snr_out_history,
                                    self.output_path)
        except Exception as exc:
            logger.warning('quality metrics figures failed: %s', exc)

    def _save_reg_breakdown_figure(
        self, reg_dict: dict[str, list[float]], epoch: int,
    ) -> None:
        """Save figure showing individual regularization terms over epochs."""
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
        epochs_x = list(range(1, len(next(iter(reg_dict.values()))) + 1))
        colors = {'L2': 'tab:blue', 'lateral_tv': 'tab:orange',
                  'depth_tv': 'tab:green', 'entropy': 'tab:red'}
        for name, values in reg_dict.items():
            ax.plot(epochs_x[:len(values)], values, '-o', markersize=3,
                    color=colors.get(name, 'k'), label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weighted regularization (gamma * term)')
        ax.set_title('Regularization breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            str(self.output_path / 'regularization_breakdown.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _compute_quality_metrics(
        self,
        model: MultiLayerModelTorch,
        epoch: int,
        hf_history: list[float],
        sbr_history: list[float],
        entropy_history: list[float],
        pupil_snr_in_history: list[float],
        pupil_snr_out_history: list[float],
    ) -> None:
        """Compute quality metrics for the current epoch and append to histories.

        Called every epoch so that metric histories are complete regardless of
        save_every. Figure saving happens separately in _save_epoch_plots.
        """
        try:
            _, out_np = self._layers_to_numpy(model)
            sample = out_np[-1]   # complex [H, W]
            pixel_size_um = float(
                self.model_params['layer_lengths'][-1]
            ) / float(self.model_params['layer_sizes'][-1])

            hf  = compute_hf_power(sample, pixel_size_um)
            sbr = compute_sbr(sample, crop_size=self.scanning_pixel_size)
            ent = compute_entropy(sample)
            hf_history.append(hf)
            sbr_history.append(sbr)
            entropy_history.append(ent)

            kin_mask_np = model.kin_mask.detach().cpu().numpy()
            p_in_np  = model.pupil_in.detach().cpu().numpy() * kin_mask_np
            p_out_np = model.pupil_out.detach().cpu().numpy() * kin_mask_np
            snr_in  = compute_pupil_snr(p_in_np)
            snr_out = compute_pupil_snr(p_out_np)
            pupil_snr_in_history.append(snr_in)
            pupil_snr_out_history.append(snr_out)

            logger.info(
                f'  Quality  HF={hf:.4f}  SBR={sbr:.2f}'
                f'  entropy={ent:.2f}  pupil_SNR in={snr_in:.1f} out={snr_out:.1f}'
            )
        except Exception as exc:
            logger.warning('quality metrics computation failed: %s', exc)

    def _median_filter_scattering_phase(
        self,
        model: MultiLayerModelTorch,
        kernel_size: int,
    ) -> None:
        """Apply a median filter to all scattering layers as complex fields,
        in-place.

        Filters Re(z) and Im(z) separately (the standard practical median
        for complex data, since complex numbers lack a total ordering).
        Affects both amplitude and phase together.

        Affects:
            - All input scattering layers (model.input_layers — N-1 entries).
            - Output scattering layers (model.output_layers[0..N-2] —
              the sample at index N-1 is NOT filtered).

        Modifies model parameters in-place; subsequent training continues
        from the filtered state.
        """
        if kernel_size <= 0:
            return
        from scipy.ndimage import median_filter
        ctype = self.complex_tensor_type

        def _filter_layer(layer_param: torch.nn.Parameter) -> None:
            with torch.no_grad():
                z = model._to_complex(layer_param, ctype).detach().cpu().numpy()
                # Median-filter the complex field: real and imag parts separately.
                z_new_re = median_filter(z.real, size=kernel_size)
                z_new_im = median_filter(z.imag, size=kernel_size)
                z_new = (z_new_re + 1j * z_new_im).astype(np.complex64)
                # Convert back to the [2, H, W] storage format used by the parameter.
                if ctype == 'Rectangular':
                    new = np.stack([z_new.real.astype(np.float32),
                                    z_new.imag.astype(np.float32)], axis=0)
                elif ctype == 'Euler':
                    new = np.stack([np.abs(z_new).astype(np.float32),
                                    np.angle(z_new).astype(np.float32)], axis=0)
                elif ctype == 'phase-only complex':
                    # Storage is [amp, phase]; complex filtering may also change
                    # the effective amplitude (channel 0) — write both channels.
                    new = np.stack([np.abs(z_new).astype(np.float32),
                                    np.angle(z_new).astype(np.float32)], axis=0)
                else:
                    raise ValueError(f"Unsupported complex_tensor_type: {ctype}")
                layer_param.copy_(torch.from_numpy(new).to(layer_param.device))

        # Input scattering layers (all of model.input_layers are scattering)
        for layer in model.input_layers:
            _filter_layer(layer)
        # Output scattering layers — exclude sample at index N-1
        n_out = len(model.output_layers)
        for k in range(max(0, n_out - 1)):
            _filter_layer(model.output_layers[k])

        logger.info(f'Median-filtered scattering layers (complex, kernel={kernel_size})')

    def _save_checkpoints(self, model: MultiLayerModelTorch, epoch: int) -> None:
        """Write .txt checkpoint files using the TF1-compatible naming convention."""
        ctype = self.complex_tensor_type
        out = str(self.output_path)

        # Pupils — native complex64; save_text handles complex via np.savetxt's
        # built-in '(re+imj)' formatting.
        p_in = model.pupil_in.detach().cpu().numpy()
        p_out = model.pupil_out.detach().cpu().numpy()
        save_text([p_in, p_out], epoch, 'pupil_aberration', out)

        # Input layers (indices 0..N-2)
        in_np = [
            _param_to_complex_np(p, ctype)
            for p in model.input_layers
        ]
        save_text(in_np, epoch, 'input_layer', out)

        # Output layers (indices 0..N-1, sample = N-1 uses Euler)
        out_np = []
        for i, p in enumerate(model.output_layers):
            ct = 'Euler' if (i == self.num_layers - 1 and ctype == 'phase-only complex') else ctype
            out_np.append(_param_to_complex_np(p, ct))
        save_text(out_np, epoch, 'output_layer', out)

        # Per-source correction maps (1D [N_sources] -> 2D [Nx, Nx])
        # Saved as source_phase_epoch_{N}.txt and source_log_amp_epoch_{N}.txt
        if getattr(model, 'use_per_source_correction', False) and hasattr(model, 'source_phase'):
            sp = model.source_phase.detach().cpu().numpy()
            sa = model.source_log_amp.detach().cpu().numpy()
            Nx = self.scanning_pixel_size
            if sp.size == Nx * Nx:
                sp_2d = sp.reshape(Nx, Nx)
                sa_2d = sa.reshape(Nx, Nx)
            else:
                # Fall back to sqrt-based reshape if sizes don't match
                side = int(np.sqrt(sp.size))
                sp_2d = sp[: side * side].reshape(side, side)
                sa_2d = sa[: side * side].reshape(side, side)
            np.savetxt(os.path.join(out, f'source_phase_epoch_{epoch + 1}.txt'), sp_2d)
            np.savetxt(os.path.join(out, f'source_log_amp_epoch_{epoch + 1}.txt'), sa_2d)

    def _layers_to_numpy(
        self, model: MultiLayerModelTorch
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (input_layers_complex, output_layers_complex) as numpy lists."""
        ctype = self.complex_tensor_type
        in_np = [_param_to_complex_np(p, ctype) for p in model.input_layers]
        out_np = []
        for i, p in enumerate(model.output_layers):
            ct = 'Euler' if (i == self.num_layers - 1 and ctype == 'phase-only complex') else ctype
            out_np.append(_param_to_complex_np(p, ct))
        return in_np, out_np


# ---------------------------------------------------------------------------
# run_from_config — entry point called by cli.py
# ---------------------------------------------------------------------------

def run_from_config(cfg: dict) -> None:
    """Execute an optimisation experiment from a config dict.

    Handles three modes: default single run, layer-number sweep, depth scan.
    """
    data_format = cfg.get('data_format', 'npy')
    illumination_dir = Path(cfg.get('illumination_dir') or '')
    init_dir = Path(cfg.get('init_dir') or cfg.get('pupil_path') or '')
    default_layer_positions = np.array(cfg['layer_positions'], dtype=np.float64)
    is_scan = cfg.get('is_scan', False)
    is_layer_number_varied = cfg.get('is_layer_number_varied', False)
    swap = cfg.get('swap', False)
    NA = cfg['NA']
    label_data_dir = Path(cfg.get('labels_data_dir') or '')
    use_amp = cfg.get('use_amp', False)
    use_compile = cfg.get('use_compile', False)

    common = dict(
        data_format=data_format,
        scanning_size=cfg.get('scanning_size', 0),
        wavelength=cfg['wavelength'],
        medium_refractive_index=cfg['medium_refractive_index'],
        resolution=cfg['resolution'],
        NA=NA,
        NA_model=cfg['NA_model'],
        geometry_mode=cfg.get('geometry_mode', 'Reflection'),
        illumination_dir=illumination_dir,
        illumination_filename=cfg.get('illumination_filename') or '',
        illumination_size=cfg.get('illumination_size', 0),
        label_data_dir=label_data_dir,
        label_filename=cfg.get('label_filename') or '',
        labels_size=cfg.get('labels_size', 0),
        ROD=cfg.get('ROD', None),
        pupil_funcs_type=cfg['pupil_funcs_type'],
        is_scan=is_scan,
        optimization_hyperparams=cfg['optimization_hyperparams'],
        TV_layer_indices=np.array(cfg.get('TV_layer_indices', [])),
        init_dir=init_dir,
        layers_init=cfg.get('layers_init', cfg.get('aberr_option', 'flat')),
        pupil_init=cfg.get('pupil_init', None),
        init_epoch=cfg.get('init_epoch', 10),
        complex_tensor_type=cfg.get('complex_tensor_type', 'Rectangular'),
        swap=swap,
        pupil_filename=cfg.get('pupil_filename') or 'pupil_aberration_*_epoch_6.txt',
        reg_mode=cfg.get('reg_mode', ['L2_norm']),
        use_amp=use_amp,
        use_compile=use_compile,
        rrmat_path=cfg.get('rrmat_path') or '',
        rrmat_sigma=cfg.get('rrmat_sigma', 0),
        rrmat_spot_offset=tuple(cfg.get('rrmat_spot_offset', (0, 0))),
        raw_mat_path=cfg.get('raw_mat_path') or '',
        raw_N_d=cfg.get('raw_N_d', 260),
        raw_xc=cfg.get('raw_xc', 129),
        raw_yc=cfg.get('raw_yc', 104),
        raw_Nx_roi=cfg.get('raw_Nx_roi', None),
        sat_repair=cfg.get('sat_repair', False),
        sat_abs_threshold=cfg.get('sat_abs_threshold', 25000),
        sat_k_sigma=cfg.get('sat_k_sigma', 6.0),
        sat_kernel_size=cfg.get('sat_kernel_size', 81),
        sat_n_passes=cfg.get('sat_n_passes', 1),
        use_per_source_correction=cfg.get('use_per_source_correction', False),
        freeze_global_phase=cfg.get('freeze_global_phase', False),
        source_amp_variable=cfg.get('source_amp_variable', True),
        illumination_mode=cfg.get('illumination_mode', cfg.get('training_mode', 'single_focus')),
        speckle=cfg.get('speckle', {}),
        intensity_loss=cfg.get('intensity_loss', {}),
        save_every=cfg.get('save_every', 1),
        phase_median_filter_kernel=cfg.get('phase_median_filter_kernel', 0),
        phase_median_filter_at_epochs=cfg.get('phase_median_filter_at_epochs', None),
    )

    # -----------------------------------------------------------------------
    # Mode: depth scan (is_scan=True)
    # Iterates over scan_range with scan_step, shifting scanning_layer_order
    # layers by dz at each point. Non-scanning layers + pupils held constant.
    # -----------------------------------------------------------------------
    if is_scan:
        scan_range = cfg.get('scan_range', [-4, 5])
        scan_step = cfg.get('scan_step', 1)
        scanning_layer_order = cfg.get('scanning_layer_order', [0])
        if not isinstance(scanning_layer_order, (list, tuple)):
            scanning_layer_order = [scanning_layer_order]
        dz_values = np.arange(scan_range[0], scan_range[1], scan_step)
        output_par_dir = Path(cfg['output_dir'])

        # Validate: scan needs preoptimized pupils + non-scanning layers.
        # Without warm-start, frozen pupils default to zero phase (meaningless).
        if not init_dir or str(init_dir) in ('', '.'):
            raise ValueError(
                'is_scan=True requires init_dir pointing to a previous full '
                'optimization output (containing pupil_aberration_*_epoch_N.txt '
                'and input/output_layer_*_epoch_N.txt). Run a full optimization '
                'first, then point init_dir to its output dir.'
            )

        logger.info(
            f'Depth scan: {len(dz_values)} points, '
            f'dz={[float(x) for x in dz_values]}, '
            f'scanning layers {scanning_layer_order}, '
            f'init_dir={init_dir}'
        )

        # Pre-load data once — all scan iterations use the same
        # illumination/labels arrays, only layer_positions change.
        preloaded_data = _load_and_process_data(
            data_format=data_format,
            rrmat_path=common.get('rrmat_path', ''),
            rrmat_sigma=common.get('rrmat_sigma', 0),
            rrmat_spot_offset=common.get('rrmat_spot_offset', (0, 0)),
            wavelength=cfg['wavelength'],
            medium_refractive_index=cfg['medium_refractive_index'],
            resolution=cfg['resolution'],
            raw_mat_path=common.get('raw_mat_path', ''),
            raw_N_d=common.get('raw_N_d', 260),
            raw_xc=common.get('raw_xc', 129),
            raw_yc=common.get('raw_yc', 104),
            raw_Nx_roi=common.get('raw_Nx_roi', None),
            sat_repair=common.get('sat_repair', False),
            sat_abs_threshold=common.get('sat_abs_threshold', 25000),
            sat_k_sigma=common.get('sat_k_sigma', 6.0),
            sat_kernel_size=common.get('sat_kernel_size', 81),
            sat_n_passes=common.get('sat_n_passes', 1),
            illumination_dir=illumination_dir,
            illumination_filename=cfg.get('illumination_filename') or '',
            illumination_size=cfg.get('illumination_size', 0),
            label_data_dir=label_data_dir,
            label_filename=cfg.get('label_filename') or '',
            labels_size=cfg.get('labels_size', 0),
            scanning_size=cfg.get('scanning_size', 0),
            ROD=cfg.get('ROD', None),
            swap=swap,
        )
        logger.info('Data pre-loaded for scan loop (will be reused across %d iterations)',
                     len(dz_values))

        for dz in dz_values:
            layer_positions = default_layer_positions.copy()
            layer_positions[scanning_layer_order] = (
                layer_positions[scanning_layer_order] + float(dz))
            layer_positions = np.round(layer_positions, 2)
            n_layers = len(layer_positions)

            # Non-scanning layers frozen (constant); scanning layers variable.
            input_obj_funcs_type = ['constant'] * n_layers
            output_obj_funcs_type = ['constant'] * n_layers
            for idx in scanning_layer_order:
                input_obj_funcs_type[idx] = 'variable'
                output_obj_funcs_type[idx] = 'variable'

            scan_pos_str = '_'.join(
                f'{float(layer_positions[idx]):.2f}'
                for idx in scanning_layer_order
            )
            output_dir = output_par_dir / f'scan_dz{float(dz):+.2f}_z{scan_pos_str}'

            logger.info(
                f'Scan dz={float(dz):+.2f}: '
                f'layer_positions={[float(x) for x in layer_positions]} '
                f'-> {output_dir}'
            )

            # Force preoptimized init for pupils + non-scanning layers.
            # The scanning layer is reset to flat via scanning_layer_order.
            # Pupils held constant (frozen at preoptimized values).
            # is_scan=False inside inner run: prevents triggering scan logic
            # again. layers_init='previous' triggers warm-start of layers,
            # then scanning_layer_order resets the scanning layer to flat.
            common_scan = dict(common)
            common_scan['pupil_funcs_type'] = ['constant', 'constant']
            common_scan['is_scan'] = False
            common_scan['layers_init'] = 'previous'
            common_scan['pupil_init'] = 'previous'
            common_scan['scanning_layer_order'] = scanning_layer_order

            _run_single(
                output_dir=output_dir,
                layer_positions=layer_positions,
                input_obj_funcs_type=input_obj_funcs_type,
                output_obj_funcs_type=output_obj_funcs_type,
                optimization_epochs=cfg.get('scan_optimization_epochs', cfg.get('scan_train_epochs', 5)),
                is_layer_number_varied=False,
                preloaded_data=preloaded_data,
                **common_scan,
            )

            # Release GPU memory between scan points to prevent fragmentation.
            # Padded sizes change with layer position, so old allocations become
            # unusable fragments. empty_cache() returns them to CUDA for reuse.
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

        # --- Auto post-process: collect layers + render video ---
        # Wrapped in try/except so post-processing failure doesn't taint
        # the scan results (which are already saved in subdirectories).
        try:
            from .scan_postprocess import make_scan_postprocess
            make_scan_postprocess(
                scan_dir=output_par_dir,
                epoch=cfg.get('scan_optimization_epochs', cfg.get('scan_train_epochs', 5)),
                pixel_size_um=cfg.get('resolution'),
                scanning_layer_index=int(scanning_layer_order[0]),
            )
        except Exception as exc:
            logger.warning('Scan post-processing failed: %s', exc)
        return

    # -----------------------------------------------------------------------
    # Default single run
    # -----------------------------------------------------------------------
    layer_positions = default_layer_positions.copy()
    n_layers = len(layer_positions)
    input_obj_funcs_type = cfg.get('input_obj_funcs_type', ['variable'] * n_layers)
    output_obj_funcs_type = cfg.get('output_obj_funcs_type', ['variable'] * n_layers)
    output_dir = Path(cfg['output_dir'])

    _run_single(
        output_dir=output_dir,
        layer_positions=layer_positions,
        input_obj_funcs_type=input_obj_funcs_type,
        output_obj_funcs_type=output_obj_funcs_type,
        optimization_epochs=cfg.get('optimization_epochs', cfg.get('train_epochs')),
        is_layer_number_varied=cfg.get('layers_init', cfg.get('aberr_option', 'flat')) == 'previous',
        **common,
    )


def _run_single(
    *,
    output_dir: Path,
    layer_positions: np.ndarray,
    input_obj_funcs_type: list[str],
    output_obj_funcs_type: list[str],
    init_dir: Path,
    optimization_epochs: int,
    is_layer_number_varied: bool,
    data_format: str = 'npy',
    scanning_size: int,
    wavelength: float,
    medium_refractive_index: float,
    resolution: float,
    NA: float,
    NA_model: float,
    geometry_mode: str,
    illumination_dir: Path = Path(''),
    illumination_filename: str = '',
    illumination_size: int = 0,
    label_data_dir: Path = Path(''),
    label_filename: str = '',
    labels_size: int = 0,
    ROD: int | None = None,
    pupil_funcs_type: list[str],
    is_scan: bool,
    optimization_hyperparams: dict,
    TV_layer_indices,
    layers_init: str,
    pupil_init: str | None = None,
    init_epoch: int = 6,
    scanning_layer_order: list | None = None,
    complex_tensor_type: str,
    swap: bool,
    pupil_filename: str,
    reg_mode,
    use_amp: bool = False,
    use_compile: bool = False,
    rrmat_path: str = '',
    rrmat_sigma: float = 0,
    rrmat_spot_offset: tuple[int, int] = (0, 0),
    raw_mat_path: str = '',
    raw_N_d: int = 260,
    raw_xc: int = 129,
    raw_yc: int = 104,
    raw_Nx_roi: int | None = None,
    sat_repair: bool = False,
    sat_abs_threshold: int = 25000,
    sat_k_sigma: float = 6.0,
    sat_kernel_size: int = 81,
    sat_n_passes: int = 1,
    use_per_source_correction: bool = False,
    freeze_global_phase: bool = False,
    source_amp_variable: bool = True,
    illumination_mode: str = 'single_focus',
    speckle: dict | None = None,
    intensity_loss: dict | None = None,
    save_every: int = 1,
    phase_median_filter_kernel: int = 0,
    phase_median_filter_at_epochs: list[int] | None = None,
    preloaded_data: dict | None = None,
) -> MultiLayerOptimizationTorch:
    """Build and run one MultiLayerOptimizationTorch experiment.

    Parameters
    ----------
    preloaded_data : dict or None
        If provided, skip data loading and use these pre-processed arrays.
        Expected keys: 'illumination_data', 'labels_data',
        'scanning_pixel_size', 'wavelength', 'refractive_index', 'resolution'.
        Produced by ``_load_and_process_data()``.  When None (default), data
        is loaded from disk as usual — this is the normal single-run path.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if preloaded_data is not None:
        # Use pre-loaded data: skip all I/O and preprocessing.
        # Override optical params in case they were derived from rrmat file.
        opt = MultiLayerOptimizationTorch(
            output_path=output_dir,
            scanning_pixel_size=preloaded_data['scanning_pixel_size'],
            layer_positions=layer_positions,
            wavelength=preloaded_data['wavelength'],
            refractive_index=preloaded_data['refractive_index'],
            resolution=preloaded_data['resolution'],
            NA=NA,
            NA_model=NA_model,
            geometry_mode=geometry_mode,
            complex_tensor_type=complex_tensor_type,
        )
        opt.illumination_data = preloaded_data['illumination_data']
        opt.labels_data = preloaded_data['labels_data']
    else:
        opt = MultiLayerOptimizationTorch(
            output_path=output_dir,
            scanning_pixel_size=scanning_size,
            layer_positions=layer_positions,
            wavelength=wavelength,
            refractive_index=medium_refractive_index,
            resolution=resolution,
            NA=NA,
            NA_model=NA_model,
            geometry_mode=geometry_mode,
            complex_tensor_type=complex_tensor_type,
        )

        if data_format == 'rrmat':
            # Pre-flight: ensure rrmat_path exists, or generate from raw data.
            from .raw2rrmat import ensure_rrmat_exists
            rrmat_path = ensure_rrmat_exists(
                rrmat_path,
                raw_mat_path=raw_mat_path or None,
                raw_N_d=raw_N_d,
                raw_xc=raw_xc,
                raw_yc=raw_yc,
                raw_Nx_roi=raw_Nx_roi,
                sat_repair=sat_repair,
                sat_abs_threshold=sat_abs_threshold,
                sat_k_sigma=sat_k_sigma,
                sat_kernel_size=sat_kernel_size,
                sat_n_passes=sat_n_passes,
            )
            Nx_det, Nx_illum = opt.load_from_rrmat(
                rrmat_path, sigma=rrmat_sigma, spot_offset=rrmat_spot_offset)

            if illumination_size not in (0, Nx_illum) and illumination_size > 0:
                logger.warning(
                    f'Config illumination_size={illumination_size} ignored in rrmat mode '
                    f'(using Nx_illum={Nx_illum} from R-matrix)')
            if labels_size not in (0, Nx_det) and labels_size > 0:
                logger.warning(
                    f'Config labels_size={labels_size} ignored in rrmat mode '
                    f'(using Nx_det={Nx_det} from R-matrix)')

            if scanning_size == 0 or scanning_size > Nx_illum:
                opt.scanning_pixel_size = Nx_illum
                logger.info(f'  scanning_size set to {Nx_illum} (from R-matrix Nx_illum)')

            if ROD is None or ROD == 0:
                ROD = opt.estimate_rod_from_rrmat()

            # Free the raw R-matrix now — illum/labels are derived and ROD
            # is known. Saves ~14 GB so apply_loose_confocal's broadcast
            # multiply doesn't push peak RAM over a tight cgroup limit.
            if hasattr(opt, '_rr_matrix'):
                del opt._rr_matrix
                import gc
                gc.collect()

            opt.indexing_input_data(swap=swap)
            opt.apply_loose_confocal(ROD)
        else:
            opt.load_illumination_data(illumination_dir, illumination_filename)
            opt.match_ROIandres_illuminations(resolution, illumination_size)
            opt.load_labels_data(label_data_dir, label_filename)
            opt.match_ROIandres_labels(resolution, labels_size)
            opt.indexing_input_data(swap=swap)
            opt.apply_loose_confocal(ROD if ROD is not None else 120)

    opt.setup_model_params()

    # Pupil initialization: pupil_init controls pupils independently from layers.
    # pupil_init='previous' loads preoptimized pupils without triggering layer warm-start.
    # Falls back to layers_init if pupil_init not set (backward compat).
    effective_pupil_init = pupil_init if pupil_init is not None else layers_init
    opt.initialize_pupil_functions(
        pupil_funcs_type,
        option=effective_pupil_init,
        path=init_dir,
        filename=pupil_filename,
        swap=swap,
    )

    # Layer initialization: only warm-start layers when layers_init='previous'
    # (not when only pupil_init='previous')
    warm_start = is_scan or is_layer_number_varied or layers_init == 'previous'
    if warm_start:
        in_types = ['constant'] * len(layer_positions) if swap else input_obj_funcs_type
        opt.initialize_object_functions(
            in_types, output_obj_funcs_type,
            mode='previous', epoch=init_epoch, indices=None,
            path=init_dir, resolution=resolution,
            scanning_layer_order=scanning_layer_order,
            swap=swap,
        )
    else:
        opt.initialize_object_functions(
            input_obj_funcs_type, output_obj_funcs_type,
            mode='initial', init_option='flat+confocal',
        )

    # Set per-source correction flag in initializer (read by model constructor)
    if use_per_source_correction:
        opt.initializer['use_per_source_correction'] = True
        # Phase-only per-source correction when source_amp_variable=False
        # (amplitude fixed at 1); read by the model constructor.
        opt.initializer['source_amp_variable'] = source_amp_variable
        # When per-source is active, optionally freeze global_phase_map
        # to avoid ambiguity between spatial and source-dependent corrections
        if freeze_global_phase:
            opt.initializer['freeze_global_phase'] = True

    opt.run_optimization(
        optimization_hyperparams, optimization_epochs,
        reg_mode=reg_mode,
        TV_layer_indices=TV_layer_indices,
        use_amp=use_amp,
        use_compile=use_compile,
        use_per_source_correction=use_per_source_correction,
        illumination_mode=illumination_mode,
        speckle=speckle if speckle else {},
        intensity_loss=intensity_loss if intensity_loss else {},
        save_every=save_every,
        phase_median_filter_kernel=phase_median_filter_kernel,
        phase_median_filter_at_epochs=phase_median_filter_at_epochs,
    )
    return opt
