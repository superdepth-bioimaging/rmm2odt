# rmm2odt — reflection matrix → 3-D ODT, one CLI

Self-contained pipeline that takes a **reflection matrix** (`rrmat.mat`) all the
way to a **3-D refractive-index (ODT) reconstruction** and a set of
publication-ready figures, combining three solvers that normally live in
separate projects plus a lean renderer. Only the code each stage actually needs
is vendored here (the shared `utils/` library is **not** a dependency).

```
rrmat.mat
  └─[1] multilayer  base reconstruction + depth scan   → output_layer_stack.npy + depths.npy
        └─[2] GVAS  synthetic angular scan             → odt_input.mat
              └─[3] BPM_ODT  FISTA + 3-D TV + BPM      → ss_opt_epochN.mat (+ final.mat)
                    └─[4] render  ortho slices + z-scan → fig_ortho.{png,svg}, ri_zscan.mp4
                          + manuscript panels           → fig2c/2e/2e/3b/3d + fig4f_*.{png,svg}
```

If you just want to **reproduce the paper figures**, follow the numbered
walkthrough below top to bottom — it takes you from a clean machine and a data
download to the six final panels.

---

## What you will produce

After a full run, `<output_root>/render/` contains:

| File | What it is |
|---|---|
| `fig2c_confocal_reflectance.{png,svg}` | Raw confocal reflectance of the sample (`diag(R)`) |
| `fig2e_optimized_layers.{png,svg}` | Per-layer reconstructed scattering layers (amp + phase) |
| `fig2e_optimized_object.{png,svg}` | Reconstructed sample plane (amp + phase) |
| `fig3b_depth_scanned_layers.{png,svg}` | Depth-scan montage of the reconstructed layer |
| `fig3d_angular_transmission_fields.{png,svg}` | Illumination + GVAS angular fields (+ propagated-rrmat rows) |
| `fig4f_multi_depth_xy_odt_images.{png,svg}` | Grayscale XY slices of the 3-D ODT volume at z = 15/20/25 µm |
| `fig_ortho.{png,svg}` | XY / XZ / YZ central slices of the ODT volume |
| `ri_zscan.mp4` | Depth sweep through the ODT volume |

---

## Step-by-step: from zero to the figures

### 1. Prerequisites

Works on **Linux and Windows** (and macOS for the CPU-only steps). Every step
below that is shell-specific gives both a **Linux / macOS (bash)** form and a
**Windows (PowerShell)** form; the `python …` commands themselves are identical
on all platforms.

- **GPU**: an NVIDIA CUDA GPU for stages 1–3 — on **either Linux or Windows**.
  The reference run used an **RTX A6000 (48 GB)**; ~10–20 GB VRAM is plenty for
  the provided config. Without a GPU you can still do `--dry-run` and (slowly)
  the render stage, but not the reconstructions.
- **System RAM**: ≥ 16 GB (the render stage loads the full ~4.7 GB `rrmat.mat`
  into memory for `fig2c` and `fig3d`).
- **Disk**: ~10 GB free (≈ 4.8 GB data + a few GB of run artifacts).
- **Software**: [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
  and `git`.
  - **Windows**: run every command below in the **Anaconda Prompt** or in
    **PowerShell** after `conda init powershell` (so `conda activate` works).
    Use forward slashes `/` in paths — Python accepts them on Windows.

### 2. Get the code

```bash
git clone https://github.com/superdepth-bioimaging/rmm2odt.git
cd rmm2odt
```

### 3. Create the Python environment

Identical on Linux and Windows (run in a terminal / Anaconda Prompt):

```bash
conda create -n rmm2odt python=3.10 -y
conda activate rmm2odt

# Install a CUDA build of PyTorch that matches your driver (see pytorch.org).
# Example (CUDA 11.8) — same command on Linux and Windows:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Everything else:
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 4. Download the data

The input data is hosted on Figshare (it is **not** in the git repo):

> **Figshare:** https://doi.org/10.6084/m9.figshare.\<ARTICLE_ID\>   ← replace with the dataset DOI

Download and unpack it into a folder of your choice. That folder must contain
the two files (names matter — the config looks them up by name):

| File | Size | Role |
|---|---|---|
| `rrmat.mat` | ~4.74 GB | the measured reflection matrix (pipeline input) |
| `279x150x150_angle_illumination.mat` | ~50 MB | the GVAS synthetic-illumination stack |

**Linux / macOS (bash):**

```bash
mkdir -p ~/rmm2odt_data
# ... download both files into ~/rmm2odt_data ...
ls -lh ~/rmm2odt_data      # expect rrmat.mat and 279x150x150_angle_illumination.mat
```

**Windows (PowerShell):**

```powershell
mkdir C:/Users/you/rmm2odt_data
# ... download both files into C:/Users/you/rmm2odt_data ...
dir C:/Users/you/rmm2odt_data   # expect rrmat.mat and 279x150x150_angle_illumination.mat
```

### 5. Point the pipeline at your data

The ready-made **`configs/reproduce_sample5.yaml`** reads its data paths from two
environment variables, so you don't have to edit any file. Set them in the
**same terminal** you'll run the pipeline from:

**Linux / macOS (bash):**

```bash
export RMM2ODT_DATA=~/rmm2odt_data                 # folder holding the two Figshare files
export RMM2ODT_OUT=~/rmm2odt_data/sample5_run      # where results go (optional; defaults to ./results/sample5_251208)
```

**Windows (PowerShell):** — use forward slashes, and set them for this session:

```powershell
$env:RMM2ODT_DATA = "C:/Users/you/rmm2odt_data"
$env:RMM2ODT_OUT  = "C:/Users/you/rmm2odt_data/sample5_run"
```

Prefer editing a file instead? Copy the config and set the three `paths:`
entries (`rrmat`, `illumination_file`, `output_root`) to absolute paths
(forward slashes are fine on Windows too).

### 6. Validate with a dry run (no GPU, nothing executed)

```bash
python run.py --config configs/reproduce_sample5.yaml --dry-run
```

This prints the full plan and **every input/output path**, marking each `OK` or
`MISSING`. Confirm the two inputs read `OK` before continuing — if `rrmat` or
`illumination_file` is `MISSING`, fix `RMM2ODT_DATA` / the download.

### 7. Run the full pipeline

```bash
python run.py --config configs/reproduce_sample5.yaml
```

- Runs all four stages in order and writes the figures at the end.
- **Budget time**: roughly one to a few hours depending on GPU (base
  reconstruction + 23-depth scan + GVAS + BPM-ODT). The final render + panels
  add a few minutes (the `fig3d` propagated-rrmat rows load the 4.7 GB matrix).
- **Keep the run alive for its full duration:**

  **Linux / macOS — on a remote server, use `tmux`/`screen`** so an SSH
  disconnect doesn't kill it, and unbuffer + log the output:

  ```bash
  tmux new -s rmm2odt
  conda activate rmm2odt
  python -u run.py --config configs/reproduce_sample5.yaml 2>&1 | tee run.log
  #   detach: Ctrl-b then d      reattach: tmux attach -t rmm2odt
  ```

  **Windows — keep the terminal window open** for the duration, and tee the
  log with PowerShell:

  ```powershell
  python -u run.py --config configs/reproduce_sample5.yaml 2>&1 | Tee-Object run.log
  ```

### 8. Collect the figures

**Linux / macOS (bash):**

```bash
ls "$RMM2ODT_OUT"/render/          # the six panels (+ fig_ortho + ri_zscan.mp4)
```

**Windows (PowerShell):**

```powershell
dir "$env:RMM2ODT_OUT/render/"     # or open the folder in File Explorer
```

All outputs are organized under `output_root/` in per-stage subfolders:
`multilayer_base/`, `multilayer_scan/`, `gvas/`, `odt/`, `render/`.

### 9. (Optional) Re-render only the figures

Stages 1–3 are the slow part. Once they've run once, you can iterate on the
figures alone (seconds–minutes, no re-optimization):

```bash
python run.py --config configs/reproduce_sample5.yaml --from render
```

`--from`/`--to` resume any contiguous subset of stages
(`multilayer | gvas | odt | render`), provided the earlier stages' artifacts
already exist under `output_root/`.

---

## Repository layout

```
rmm2odt/
  run.py              # CLI entry (adds vendored pkgs to sys.path)
  rmm_pipeline.py     # orchestrator: chains stages, passes artifacts, --from/--to, --dry-run
  rmm_config.py       # one unified YAML → each stage's native config
  rmm_render.py       # stage 4: ortho slices (XY/XZ/YZ) + z-scan video (matplotlib only)
  rmm_panels.py       # stage 4b: the six manuscript panels, pruned & self-contained
  rmm_rm.py           # numpy-only reflection-matrix toolchain for fig3d's propagated-rrmat rows
  _common/            # the 2 utils helpers the renderer needs (scalebar, colorbar)
  multilayer_torch/   # vendored+pruned stage 1  (entry: solver.run_from_config)
  gvas_torch/         # vendored+pruned stage 2  (entry: solver.run_optimization)
  odt/                # vendored+pruned stage 3  (entry: reconstruct.reconstruct)
  configs/
    reproduce_sample5.yaml     # portable, env-var paths — start here
    sample5_251208_medium.yaml # the original absolute-path config for this sample
    sample5_251208.yaml        # faster/lower-quality "shakedown" variant
    example_pipeline.yaml      # annotated template
  requirements.txt
```

## The pipeline stages

| # | Stage | Entry point | Produces |
|---|---|---|---|
| 1 | multilayer | `multilayer_torch.solver.run_from_config` | base recon layers (`multilayer_base/`) + depth-scan stack `multilayer_scan/output_layer_stack.npy` + `depths.npy` |
| 2 | GVAS | `gvas_torch.solver.run_optimization` | `gvas/odt_input.mat` (`Illumination_stack` / `Output_stack`) |
| 3 | BPM_ODT | `odt.reconstruct.reconstruct` | `odt/ss_opt_epochN.mat` (+ `final.mat`), the 3-D Δn volume |
| 4 | render | `rmm_render` + `rmm_panels` | `render/` figures + z-scan video |

## Manuscript panels (stage 4b)

Rendered straight from the run's own artifacts, matching the published panels:

| Panel file | Built from |
|---|---|
| `fig2c_confocal_reflectance` | `paths.rrmat` (confocal = `diag(R)`) |
| `fig2e_optimized_layers` | `multilayer_base/output_layer_*_epoch_{base_epochs}.txt` (layers 1..N-1) |
| `fig2e_optimized_object` | same dir, last layer (sample plane) |
| `fig3b_depth_scanned_layers` | `multilayer_scan/output_layer_stack.npy` + `depths.npy` |
| `fig3d_angular_transmission_fields` | `gvas/odt_input.mat` (+ optional propagated `rrmat`) |
| `fig4f_multi_depth_xy_odt_images` | `odt/ss_opt_epochN.mat` — **grayscale, z = 15/20/25 µm** by default |

Enabled by default. Toggle / tune via the `render.panels` block:

```yaml
render:
  panels:
    enable: true
    formats: [png, svg]
    fig4_z_um: [15, 20, 25]   # XY depths for fig4f (grayscale)
    fig4_cmap: gray
    crop_2d: 156              # fig2e_optimized_object crop (null = adaptive)
    # object_size_1b: 120     # fig2c_confocal_reflectance crop
    # angle_indices: [0,5,15] # fig3d angles (null = 5 evenly spaced)
    fig3d_rrmat: true              # add fig3d rows: angular image from the reflection matrix
    fig3d_rrmat_propagate_d: 14    # propagate the rrmat by this many µm (ASM) first
    # fig3d_rrmat_NA: null         # optional aperture NA for the propagation cone
```

**fig3d with `fig3d_rrmat`** grows from 3 rows (illum. phase / GVAS |E| / GVAS
phase) to **5 rows**, adding the angular image extracted from the reflection
matrix — propagated by `fig3d_rrmat_propagate_d` µm — at the same illumination
k-vector (rows 4–5 = `Rxk |E|` / `Rxk phase`). This uses the vendored
`rmm_rm.py` (numpy-only pruned port of the `utils/` reflection-matrix toolchain:
`symmetrize_rm` / `rm_to_xk` / `propage_rrmat`); loading the full rrmat is
memory-heavy (several GB).

Each panel is guarded — a missing input or a failing panel is
warned-and-skipped, not fatal.

## Notes / conventions carried over from the source projects

- **Stage 1 is two passes.** The depth scan warm-starts from a full
  reconstruction. Leave `multilayer.init_dir` empty to run the base
  reconstruction first; set it to an existing optimisation dir to skip the base
  pass and scan from it.
- **The scanning layer is auto-derived from the scan window.** You define
  `layer_positions` and an **absolute** `scan_range: [z_lo, z_hi]` (µm):
  - **Exactly one** layer inside → that layer is swept; all others frozen.
  - **Two or more** inside → the base recon keeps the **full** model, but the
    depth scan **auto-collapses** the in-window layers into **ONE object layer**
    (`scan_collapse_mode` = `center` | `bottom` | `mean`) and sweeps it; outside
    layers are frozen (carried via a staged reduced-model init-dir).
  - **Zero** inside → error.
- **Geometry/dtype gotchas** are preserved from the originals: R-matrix `rr.T` +
  F-order reshape (stage 1); `Illumination_stack`/`Output_stack` names and
  per-image CASS normalisation (stage 2); `(Ny, Nx, Nz)` array shape and the
  `N//2+1` MATLAB-centering convention (stage 3).
- **GPU.** Stages 1–3 need a CUDA torch env for realistic data sizes.
  `--dry-run` needs neither torch nor GPU.

## Troubleshooting

- **`MISSING rrmat` / `MISSING illumination_file` in `--dry-run`** — `RMM2ODT_DATA`
  isn't set or the two files aren't in it with the exact expected names.
- **`torch.cuda.is_available()` is False** — you installed a CPU-only PyTorch.
  Reinstall a CUDA build matching your driver (see step 3).
- **Out-of-memory during render (`fig2c`/`fig3d`)** — these load the full
  ~4.7 GB rrmat; the `fig3d` propagated-rrmat step needs several GB more. Free
  RAM, or set `render.panels.fig3d_rrmat: false` to skip the heavy rows.
- **No `.mp4`, only `.gif`** — `ffmpeg` isn't available; `pip install
  imageio-ffmpeg` (already in `requirements.txt`) or install system `ffmpeg`.
- **Nicer scale bars** — `pip install matplotlib-scalebar` (optional; the
  renderer falls back to a drawn bar without it).
- **Resume after a crash** — rerun with `--from <stage>`; completed stages'
  artifacts under `output_root/` are reused.

## Data & code availability

- **Code**: this repository (GitHub).
- **Data**: Figshare, DOI above. `rrmat.mat` + `279x150x150_angle_illumination.mat`.

If you use this pipeline, please cite the associated paper and the Figshare
dataset. *(Add the citation / BibTeX here before publishing.)*
