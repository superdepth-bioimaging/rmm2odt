"""
multilayer_torch/model.py
--------------------------
MultiLayerModelTorch — PyTorch nn.Module for the Reflection geometry
multi-layer scattering reconstruction.

Direct port of MultiLayerModel_v5 (Reflection branch) from
  multilayer_model_tensorflow1.x/multilayer/tf_models/multilayer.py

Key differences from TF1:
  - All H (transfer function) filters are precomputed once at __init__ and
    stored as non-learnable GPU buffers → no redundant FFT-grid computation
    inside forward().
  - Layer parameters are stored as 2-channel float32 [2, H, W] (re / im) to
    keep .txt checkpoint files bit-identical with TF1.x. Complex conversion
    happens inside forward() only.
  - Native torch.complex64 arithmetic instead of manual 2-channel TF ops.
  - Batch FFT: torch.fft.fft2 operates over the whole batch in one cuFFT call.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

from .physics import (
    change_size,
    fft2c,
    ifft2c,
    np_propagation_filter,
    np_pupil_mask,
    propagate,
)

# Gaussian blur applied to the pupil mask edge.
# Prof's v8 uses cv2.GaussianBlur((21,21), 0) which gives sigma ~3.5.
_PUPIL_BLUR_SIGMA = 3.5


class MultiLayerModelTorch(nn.Module):
    """Reflection geometry multi-layer scattering model.

    Parameters
    ----------
    model_params : dict
        Output of MultiLayerOptimizationTorch.setup_model_params().
        Required keys: illumination_size, size, wavelength, NA, propagatedNA,
        # Layers, layer_positions, separate_distances, depth,
        medium refractive index, layer_sizes, layer_lengths.
    initializer : dict
        Output of MultiLayerOptimizationTorch.initialize_*() methods.
        Required keys: input_object_functions, output_object_functions,
        pupil_functions, pupil_funcs_type, input_obj_funcs_type,
        output_obj_funcs_type, complex_tensor_type.
    """

    def __init__(self, model_params: dict, initializer: dict) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Store optical / geometric scalars                                   #
        # ------------------------------------------------------------------ #
        self.illumination_size: int = model_params['illumination_size']
        self.labels_size: int = model_params['size']
        self.lmb: float = model_params['wavelength']          # raw λ (same units as layer_positions)
        self.NA: float = model_params['NA']
        self.NA_prop: float = model_params['propagatedNA']
        self.N: int = model_params['# Layers']
        self.layer_positions: np.ndarray = np.asarray(model_params['layer_positions'])
        self.delta_z: np.ndarray = np.asarray(model_params['separate_distances'])
        self.d: float = float(model_params['depth'])
        self.n: float = float(model_params['medium refractive index'])
        self.padded_sizes: np.ndarray = np.asarray(model_params['layer_sizes'], dtype=int)
        self.extended_Ls: np.ndarray = np.asarray(model_params['layer_lengths'])
        self.complex_tensor_type: str = initializer['complex_tensor_type']

        lmb_eff = self.lmb / self.n   # effective wavelength λ/n in medium

        # ------------------------------------------------------------------ #
        # Learnable parameters                                                #
        # ------------------------------------------------------------------ #
        # input_object_functions has N entries; indices 0..N-2 are used in
        # the forward (illumination) path before the sample plane.
        input_funcs = initializer['input_object_functions']    # list[ndarray [2,H,W]], len N
        output_funcs = initializer['output_object_functions']  # list[ndarray [2,H,W]], len N
        pupil_funcs = initializer['pupil_functions']           # [pupil_in, pupil_out], each [2,H,W]
        pupil_types = initializer['pupil_funcs_type']          # ['variable'|'constant', ...]
        in_types = initializer['input_obj_funcs_type']         # list[str], len N
        out_types = initializer['output_obj_funcs_type']       # list[str], len N

        # Input scattering layers: indices 0..N-2
        self.input_layers = nn.ParameterList()
        for arr, typ in zip(input_funcs[: self.N - 1], in_types[: self.N - 1]):
            self.input_layers.append(
                nn.Parameter(
                    torch.from_numpy(np.array(arr, dtype=np.float32)),
                    requires_grad=(typ == 'variable'),
                )
            )

        # Output scattering layers: indices 0..N-1 (index N-1 = sample plane)
        self.output_layers = nn.ParameterList()
        for arr, typ in zip(output_funcs, out_types):
            self.output_layers.append(
                nn.Parameter(
                    torch.from_numpy(np.array(arr, dtype=np.float32)),
                    requires_grad=(typ == 'variable'),
                )
            )

        # Pupil aberration functions — stored as native complex64 [H, W].
        # The initialiser produces [2, H, W] float32 arrays (re, im channels) for
        # checkpoint compatibility; we collapse them to complex64 here.
        def _to_complex_param(arr2ch: np.ndarray) -> torch.Tensor:
            arr = np.array(arr2ch, dtype=np.float32)
            return torch.from_numpy((arr[0] + 1j * arr[1]).astype(np.complex64))

        self.pupil_in = nn.Parameter(
            _to_complex_param(pupil_funcs[0]),
            requires_grad=(pupil_types[0] == 'variable'),
        )
        self.pupil_out = nn.Parameter(
            _to_complex_param(pupil_funcs[1]),
            requires_grad=(pupil_types[1] == 'variable'),
        )

        # Global phase map (one scalar per illumination pixel, no batch dim)
        # When per-source correction is active, global_phase_map can optionally
        # be frozen to avoid ambiguity between spatial and source-dependent corrections.
        freeze_global_phase = initializer.get('freeze_global_phase', False)
        self.global_phase_map = nn.Parameter(
            torch.zeros(self.illumination_size, self.illumination_size),
            requires_grad=(not freeze_global_phase),
        )

        # Per-source phase/amplitude correction (optional).
        # n_sources = scanning_size² = actual number of sources after
        # indexing_input_data (NOT illumination_size² which is the spatial
        # image size — they differ when scanning_size != Nx_illum in rrmat mode).
        self.use_per_source_correction = initializer.get('use_per_source_correction', False)
        if self.use_per_source_correction:
            scanning_size = int(model_params.get('scanning_size', self.illumination_size))
            n_sources = scanning_size * scanning_size
            self.source_phase = nn.Parameter(torch.zeros(n_sources, dtype=torch.float32))
            # source_amp_variable=False -> per-source amplitude is fixed at 1
            # (log_amp == 0, non-learnable buffer); only the phase is optimised.
            if initializer.get('source_amp_variable', True):
                self.source_log_amp = nn.Parameter(torch.zeros(n_sources, dtype=torch.float32))
            else:
                self.register_buffer('source_log_amp',
                                     torch.zeros(n_sources, dtype=torch.float32))

        # ------------------------------------------------------------------ #
        # Precomputed non-learnable buffers                                   #
        # ------------------------------------------------------------------ #
        sz0 = int(self.padded_sizes[0])
        L0 = float(self.extended_Ls[0])

        # Pupil mask: binary circle in k-space, softened with Gaussian blur.
        # Uses measurement NA (not propagation NA).
        kin_mask_np = np_pupil_mask(sz0, lmb_eff, L0, self.NA)
        kin_mask_np = gaussian_filter(kin_mask_np, sigma=_PUPIL_BLUR_SIGMA)
        self.register_buffer('kin_mask', torch.from_numpy(kin_mask_np.astype(np.float32)))

        # NA mask for pupil gradient masking: [sz0, sz0] float32 (broadcasts
        # over the complex64 pupil gradient at usage time).
        self.register_buffer('pupil_na_mask', torch.from_numpy(kin_mask_np.astype(np.float32)))

        # H_surface: ASM filter for initial propagation (surface → deepest layer)
        # z = -d because layer_positions[0] is below the objective at +d.
        H_surf = np_propagation_filter(sz0, -self.d, lmb_eff, L0, self.NA_prop)
        self.register_buffer('H_surface', torch.from_numpy(H_surf))

        # H_surface_bw: ASM filter for back-propagating the measurement labels.
        H_surf_bw = np_propagation_filter(sz0, self.d, lmb_eff, L0, self.NA_prop)
        self.register_buffer('H_surface_bw', torch.from_numpy(H_surf_bw))

        # H_layer_k: ASM filter for inter-layer propagation at each depth step.
        # Both the forward path (propagate then crop) and backward path
        # (pad then propagate) use the same delta_z[k] and padded_sizes[k],
        # so a single buffer per k is sufficient.
        for k in range(self.N - 1):
            sz_k = int(self.padded_sizes[k])
            L_k = float(self.extended_Ls[k])
            dz_k = float(self.delta_z[k])
            H_k = np_propagation_filter(sz_k, dz_k, lmb_eff, L_k, self.NA_prop)
            self.register_buffer(f'H_layer_{k}', torch.from_numpy(H_k))

        # ------------------------------------------------------------------ #
        # Light cone layer masks                                              #
        # ------------------------------------------------------------------ #
        # Circular masks per layer based on light cone geometry.
        # Scattering layers: radius = FOV_radius * margin + depth_spread * margin
        # Sample layer (last): radius = round(sqrt(2) * scanning_size) / 2
        # Gradients outside masks are zeroed after backward().
        scanning_size = model_params.get('scanning_size', self.illumination_size)
        self._create_layer_masks(scanning_size)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _to_complex(self, param: torch.Tensor, ctype: str | None = None) -> torch.Tensor:
        """Convert a [2, H, W] float32 parameter to a [H, W] complex64 tensor.

        Dispatch table
        --------------
        'Rectangular'       : re + i·im         (general complex layer)
        'Euler'             : A · exp(i·φ)       (amplitude × phase)
        'phase-only complex': exp(i·φ)           (unit-amplitude phase mask)

        The sample plane uses 'Euler' even when the global type is
        'phase-only complex' — this retains the amplitude channel so the
        reconstructed object keeps physical meaning.
        """
        if ctype is None:
            ctype = self.complex_tensor_type
        if ctype == 'Rectangular':
            return torch.complex(param[0], param[1])
        elif ctype == 'Euler':
            return param[0] * torch.exp(1j * param[1])
        elif ctype == 'phase-only complex':
            return torch.exp(1j * param[1])
        else:
            raise ValueError(
                f"Unknown complex_tensor_type '{ctype}'. "
                "Must be 'Rectangular', 'Euler', or 'phase-only complex'."
            )

    def _h_layer(self, k: int) -> torch.Tensor:
        """Return the precomputed H buffer for inter-layer index k."""
        return getattr(self, f'H_layer_{k}')

    def _create_layer_masks(self, scanning_size: int) -> None:
        """Create circular masks for each layer based on light cone geometry.

        Scattering layers (0..N-2): mask radius = FOV_radius * margin + depth_spread * margin
        Sample layer (N-1): mask diameter = round(sqrt(2) * scanning_size)

        Pixels outside the mask have their gradients zeroed after backward(),
        preventing the optimizer from learning non-physical structure in the
        padded dark corners.
        """
        theta = np.arcsin(min(self.NA, self.n) / self.n)
        fov_radius = self.illumination_size / 2
        margin = 1.2  # 20% safety margin
        pixel_size = self.lmb / (2 * self.NA)  # resolution in um

        for i in range(self.N):
            z = float(self.layer_positions[i])
            ps = int(self.padded_sizes[i])
            is_sample = (i == self.N - 1)

            if is_sample:
                # Sample layer: mask diameter = round(sqrt(2) * scanning_size)
                mask_radius = round(np.sqrt(2) * scanning_size) / 2
            else:
                # Scattering layers: light cone spread
                spread = z * np.tan(theta) / pixel_size  # spread in pixels
                mask_radius = fov_radius * margin + spread * margin

            mask_radius = min(mask_radius, ps / 2)

            cy, cx = ps / 2, ps / 2
            yy, xx = np.mgrid[0:ps, 0:ps]
            circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= mask_radius ** 2
            mask = torch.from_numpy(circle.astype(np.float32))
            self.register_buffer(f'_layer_mask_{i}', mask)

            n_active = int(mask.sum().item())
            logger.info(
                f'  Layer {i} (z={z:.0f} um, {ps}x{ps}): '
                f'{n_active}/{ps*ps} pixels active ({100*n_active/(ps*ps):.0f}%)'
                f'{"  [sample]" if is_sample else ""}'
            )

    def mask_gradients(self) -> None:
        """Zero gradients outside physical apertures for pupils and layers.

        Call after loss.backward() and before optimizer.step().
        - Pupils: masked by NA aperture (kin_mask)
        - Layers: masked by light cone (circular masks per depth)
        """
        # Pupil NA mask: float32 [sz0, sz0]; cast to grad dtype (complex64)
        # so in-place mul_ accepts it without dtype-mismatch error.
        na_mask = self.pupil_na_mask
        if self.pupil_in.grad is not None:
            self.pupil_in.grad.mul_(na_mask.to(self.pupil_in.grad.dtype))
        if self.pupil_out.grad is not None:
            self.pupil_out.grad.mul_(na_mask.to(self.pupil_out.grad.dtype))

        # Layer light cone masks
        for i in range(self.N - 1):
            # Input layers (0..N-2)
            if i < len(self.input_layers) and self.input_layers[i].grad is not None:
                mask = getattr(self, f'_layer_mask_{i}')  # [H, W]
                self.input_layers[i].grad.mul_(mask)      # broadcasts over [2, H, W]
        for i in range(self.N):
            # Output layers (0..N-1, including sample)
            if i < len(self.output_layers) and self.output_layers[i].grad is not None:
                mask = getattr(self, f'_layer_mask_{i}')
                self.output_layers[i].grad.mul_(mask)

    # Keep old name as alias for backward compatibility
    mask_pupil_gradients = mask_gradients

    # ---------------------------------------------------------------------- #
    # Forward pass — Reflection geometry                                     #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        source_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reflection geometry forward pass.

        Parameters
        ----------
        data_x : complex64 Tensor [B, illum_H, illum_W]
            Measured illumination fields (one per scanning position in batch).
        data_y : complex64 Tensor [B, label_H, label_W]
            Measured back-scattered label fields.
        source_idx : int Tensor [B], optional
            Per-batch source indices for per-source phase/amplitude correction.
            Only used when use_per_source_correction=True.

        Returns
        -------
        output_imgs : complex64 Tensor [B, padded_sizes[0], padded_sizes[0]]
            Forward model prediction (filtered output field at objective).
        bw_gt : complex64 Tensor [B, padded_sizes[0], padded_sizes[0]]
            Back-propagated measurement (ground truth for Pearson loss).
        """
        sz0 = int(self.padded_sizes[0])

        # ------------------------------------------------------------------ #
        # 1.  Apply global phase correction to illumination                   #
        # ------------------------------------------------------------------ #
        # global_phase_map is float32 [illum_H, illum_W]; broadcast over batch.
        waves = data_x * torch.exp(1j * self.global_phase_map.to(data_x.dtype))

        # Per-source phase/amplitude correction (when enabled and indices provided)
        if self.use_per_source_correction and source_idx is not None:
            sp = self.source_phase[source_idx]       # (B,)
            sa = self.source_log_amp[source_idx]     # (B,)
            correction = torch.exp(sa + 1j * sp).to(waves.dtype)  # (B,) complex
            waves = waves * correction[:, None, None]

        # ------------------------------------------------------------------ #
        # 2.  Pad to padded_sizes[0] and apply pupil_in × kin_mask           #
        # ------------------------------------------------------------------ #
        waves = change_size(waves, sz0)

        # kin_mask: float32 [sz0, sz0] → complex64 for multiplication.
        # pupil_in is now a native complex64 [sz0, sz0] parameter; used directly.
        kin_c = self.kin_mask.to(waves.dtype)
        pupil_in_c = self.pupil_in
        waves = ifft2c(fft2c(waves) * kin_c * pupil_in_c)

        # ------------------------------------------------------------------ #
        # 3.  Propagate from objective to deepest (first) scattering layer    #
        # ------------------------------------------------------------------ #
        waves = propagate(waves, self.H_surface)   # z = -d

        # ------------------------------------------------------------------ #
        # 4.  Apply first input scattering layer (index 0)                   #
        # ------------------------------------------------------------------ #
        waves = waves * self._to_complex(self.input_layers[0])

        # ------------------------------------------------------------------ #
        # 5.  Main loop: forward path → sample plane → backward path         #
        # ------------------------------------------------------------------ #
        # Loop runs for i = 1 .. 2*N-2 (total 2*N-2 steps).
        #
        # i < N-1          : input layers 1..N-2 (forward, deeper → shallower)
        # i == N-1         : sample plane (deepest output layer, index N-1)
        # i > N-1 (i≤2N-2) : output layers N-2..0 (backward, shallower → surface)
        #
        # Index mapping for backward path:  k = 2*N-2-i
        #   i=N   → k=N-2 (second-deepest layer)
        #   i=2N-2 → k=0  (shallowest layer, back at surface)

        for i in range(1, 2 * self.N - 1):

            if i < self.N - 1:
                # --- Forward input layer ---
                # Propagate in the current (deeper) padded size, then crop
                # to the next (shallower, smaller) padded size.
                H_k = self._h_layer(i - 1)
                waves = change_size(propagate(waves, H_k), int(self.padded_sizes[i]))
                waves = waves * self._to_complex(self.input_layers[i])

            elif i == self.N - 1:
                # --- Sample plane ---
                # Same propagate-then-crop as forward layers.
                H_k = self._h_layer(i - 1)
                waves = change_size(propagate(waves, H_k), int(self.padded_sizes[i]))
                # Sample plane: always use 'Euler' for 'phase-only complex' mode
                # so both amplitude and phase channels carry information.
                sample_ctype = (
                    'Euler'
                    if self.complex_tensor_type == 'phase-only complex'
                    else self.complex_tensor_type
                )
                waves = waves * self._to_complex(self.output_layers[self.N - 1], sample_ctype)

            else:
                # --- Backward output layer ---
                k = 2 * self.N - 2 - i  # layer index (N-2 down to 0)
                # Pad to the larger (shallower) padded size, then propagate.
                waves = propagate(change_size(waves, int(self.padded_sizes[k])), self._h_layer(k))
                waves = waves * self._to_complex(self.output_layers[k])

        # ------------------------------------------------------------------ #
        # 6.  Apply output pupil at the objective plane                       #
        # ------------------------------------------------------------------ #
        # Apply kin_mask to output pupil to enforce NA constraint on detection.
        # pupil_out is native complex64; used directly.
        pupil_out_c = self.pupil_out
        output_imgs = ifft2c(fft2c(waves) * kin_c * pupil_out_c)

        # ------------------------------------------------------------------ #
        # 7.  Back-propagate measurement labels (ground truth)               #
        # ------------------------------------------------------------------ #
        # Pad labels to padded_sizes[0] and propagate upward by +d so the
        # comparison is in the same (objective) plane as output_imgs.
        bw_gt = propagate(change_size(data_y, sz0), self.H_surface_bw)

        return output_imgs, bw_gt

    # ---------------------------------------------------------------------- #
    # Convenience: collect learnable complex tensors for regularisation       #
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # Forward pass — Model B (sample-plane fields, phase-only views)         #
    # ---------------------------------------------------------------------- #

    def forward_sample_plane_fields(
        self,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        source_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute illumination and back-propagated fields at the sample plane.

        Used by Model B (coherent intensity loss). Shares all learnable
        parameters with `forward()` but applies them as **phase-only**
        operators:
            - pupils:           exp(i·φ_pupil)
            - input layers:     exp(i·φ_in_layer)
            - output layers:    exp(i·φ_out_layer)   (excluding sample)

        The sample layer (output_layers[N-1]) is NOT applied to either
        field, so cost_B contributes no gradient to the sample parameter
        (by design — Model B is sample-implicit).

        Returns
        -------
        E_in_at_sample : complex64 [B, padded_sizes[N-1], padded_sizes[N-1]]
            Illumination field arriving at the sample plane (pre-sample).
        E_back_at_sample : complex64 [B, padded_sizes[N-1], padded_sizes[N-1]]
            Measurement field back-propagated to the sample plane (post-sample).
        """
        sz0 = int(self.padded_sizes[0])
        sz_sample = int(self.padded_sizes[self.N - 1])

        # Phase-only views of the (now complex) pupil parameters: exp(i·angle).
        # Both pupils stay learnable from cost_B; pole is closed at loss level
        # by mean-based Wiener ε.
        # Selective detachment (e.g. per-source amplitude vs phase) for cost_B
        # is performed by the caller when constructing data_x — see solver.py
        # speckle path. This method itself does no detachment so it remains
        # symmetric with forward() and reusable.
        eps = 1e-8
        pupil_in_phase = self.pupil_in / (self.pupil_in.abs() + eps)
        pupil_out_phase = self.pupil_out / (self.pupil_out.abs() + eps)

        kin_c = self.kin_mask.to(data_x.dtype)

        # ===================================================================
        # E_in_at_sample : forward path up to (excluding) sample multiplication
        # ===================================================================
        # 1. Global phase + per-source corrections (same as forward()).
        waves = data_x * torch.exp(1j * self.global_phase_map.to(data_x.dtype))
        if self.use_per_source_correction and source_idx is not None:
            sp = self.source_phase[source_idx]
            sa = self.source_log_amp[source_idx]
            correction = torch.exp(sa + 1j * sp).to(waves.dtype)
            waves = waves * correction[:, None, None]

        # 2. Pad and apply phase-only input pupil + NA mask
        waves = change_size(waves, sz0)
        waves = ifft2c(fft2c(waves) * kin_c * pupil_in_phase)

        # 3. Propagate to deepest input scattering layer
        waves = propagate(waves, self.H_surface)

        # 4. Apply input_layers[0] (phase-only)
        waves = waves * self._to_complex(self.input_layers[0], 'phase-only complex')

        # 5. Forward through input_layers[1..N-2], then propagate to sample plane
        # i = 1..N-2: standard input-layer steps
        # i = N-1   : propagate + crop to sample plane size; STOP before sample mult
        for i in range(1, self.N):
            H_k = self._h_layer(i - 1)
            waves = change_size(propagate(waves, H_k), int(self.padded_sizes[i]))
            if i < self.N - 1:
                waves = waves * self._to_complex(
                    self.input_layers[i], 'phase-only complex'
                )
            # i == N-1: arrived at sample plane, no multiplication

        E_in_at_sample = waves  # [B, sz_sample, sz_sample]

        # ===================================================================
        # E_back_at_sample : invert measurement back to sample plane
        # ===================================================================
        # bw_gt has the same plane as forward()'s output_imgs (post H_surface_bw,
        # at sz0). To reach sample plane we undo: pupil_out, then the entire
        # backward-half of forward() (output layers k=0..N-2) in reverse
        # temporal order. Sample layer is NOT undone.
        bw = propagate(change_size(data_y, sz0), self.H_surface_bw)

        # 1. Undo phase-only output pupil (multiply by conjugate, apply NA mask)
        bw = ifft2c(fft2c(bw) * kin_c * pupil_out_phase.conj())

        # 2. Reverse output stack: forward did (k=N-2 → 0); inversion is k=0 → N-2.
        # Forward step k:  change_size_up(padded_sizes[k]) → propagate(H_k) → × output_layers[k]
        # Inverse step k:  × conj(output_layers[k]) → propagate(conj(H_k)) → change_size_down(padded_sizes[k+1])
        for k in range(self.N - 1):
            out_layer_phase = self._to_complex(
                self.output_layers[k], 'phase-only complex'
            )
            bw = bw * out_layer_phase.conj()
            H_k = self._h_layer(k)
            bw = propagate(bw, H_k.conj())
            bw = change_size(bw, int(self.padded_sizes[k + 1]))

        E_back_at_sample = bw  # [B, sz_sample, sz_sample]

        return E_in_at_sample, E_back_at_sample

    # ---------------------------------------------------------------------- #
    # Convenience: collect learnable complex tensors for regularisation       #
    # ---------------------------------------------------------------------- #

    def named_complex_layers(self) -> list[tuple[str, torch.Tensor, int]]:
        """Return (name, complex_tensor, size) for all learnable layer parameters.

        Used by loss.py regularisation functions which need complex [H, W] tensors.
        Only includes layers where requires_grad=True.

        Returns
        -------
        list of (name, complex Tensor [H, W], size int)
        """
        result = []
        for i, p in enumerate(self.input_layers):
            if p.requires_grad:
                result.append((
                    f'input_layer_{i}',
                    self._to_complex(p),
                    int(self.padded_sizes[i]),
                ))
        for i, p in enumerate(self.output_layers):
            if p.requires_grad:
                ctype = (
                    'Euler'
                    if (i == self.N - 1 and self.complex_tensor_type == 'phase-only complex')
                    else self.complex_tensor_type
                )
                result.append((
                    f'output_layer_{i}',
                    self._to_complex(p, ctype),
                    int(self.padded_sizes[i]),
                ))
        return result
