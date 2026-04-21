# microsimflow

A Python-based workflow integrating custom 3D microstructure modeling with property evaluations (thermal, electrical, and mechanical) using the `chfem` and `PuMA` solvers.

This repository provides an end-to-end flat script structure—from structure generation to solver execution and result visualization. It features robust experiment management, allowing you to reproduce and expand large-scale parameter sweeps without managing complex directory hierarchies or worrying about data loss.

Each run writes a CSV log that stores both solver outputs and lightweight structure descriptors computed directly from the generated `final_grid`. These descriptors are also available in `--recalc` mode, so legacy models can be re-evaluated without rebuilding geometry.

## ✨ Key Features
* **Diverse Background Phases**: Single phase, Gyroid (co-continuous), Lamellar, Cylinder, BCC, Sea-Island structures.
* **Adaptive Filler Placement**: Rigid spheres, flakes, rigid cylinders, irregular-cross-section extruded fibers (`irregfiber`), and topology-adaptive flexible fibers/agglomerates.
* **Orientation-Controlled Fillers**: Supports von Mises-Fisher (vMF) orientation control for rigid fibers, flakes, and irregular-cross-section extruded fibers via `mean_dir=ax,ay,az` and `kappa`.
* **Kinematic Deformation**: Simulate affine mechanical stretching with dynamic rigid-body kinematics for rigid fillers, outputting robust VTM/PVD time-series collections for ParaView.
* **Dual Solver Integration**: 
  * `chfem`: High-efficiency homogenization solver with GPU support. *(Note: This environment compiles a custom fork to enable the specific build options required for models with 5 or more properties).*
  * `PuMA`: Multipurpose solver for the Laplace equation.
* **Robust Experiment Management**: Automated sequential directory creation (`result_expX_01/`, `02/`...) prevents data overwriting, while all metrics are safely appended to a central `.csv` log with real-time disk syncing.
* **Cloud & Local Ready**: Includes a `Dockerfile` for local GPU workstations and a Jupyter Notebook for instant execution on Google Colab (T4 GPU).

## 📦 Dependencies
`microsimflow` relies on several high-performance scientific Python libraries and two external physics solvers.

**Python Libraries (see `requirements.txt`):**
* **Core**: `numpy`, `scipy`, `pandas`
* **Acceleration**: `numba` (JIT compilation for ultra-fast voxel collision detection and RSA placement)
* **Visualization & Plotting**: 
  * `pyvista`: Exporting standardized 3D visual data (`.vti`, `.vtm`) for ParaView.
  * `matplotlib`, `seaborn`: Generating statistical charts.
  * `pillow`: Image processing for 2D visual montages.
* **Utilities**: `tqdm` (CLI progress), `jupyterlab` (Notebook execution)

**External Solvers:**
* **`chfem`**: High-performance computational homogenization (Compiled from C/CUDA).
* **`PuMA`**: NASA's Porous Microstructure Analysis software.

---

## 🚀 Environment Setup

We **strongly recommend** using the provided `Dockerfile` to build the environment. 
Because this workflow relies on specialized physics solvers (`PuMA` via Conda, and a custom CUDA compilation of `chfem`), a standard `pip install -r requirements.txt` will not fully set up the solvers.

### Option A: Local Run via Docker (Recommended)
This process installs all Python dependencies, `PuMA`, and **compiles `chfem` from source** (CUDA 12.x). 

> **⚠️ Important note on `chfem`:** > Models with 5 or more properties require the `-DCUDAPCG_MATKEY_32BIT` flag during compilation. Because the official `chfem` repository currently fails to build when this flag is passed, our Dockerfile automatically clones and compiles a **temporarily forked version** that contains the necessary fixes. 
> The Dockerfile also uses `-DCMAKE_CUDA_ARCHITECTURES=native` to automatically optimize the solver for your local GPU.

```bash
# Build the Docker image
docker build -t microsim_env .

# Run the container (Requires NVIDIA Container Toolkit)
docker run -it --rm --gpus all -v $(pwd):/workspace microsim_env bash
````

### Option B: Google Colab (Cloud)

If you do not have a local GPU workstation, you can use the provided `microsimflow_on_colab.ipynb`.
Upload the notebook to Google Colab, set the runtime to **T4 GPU**, and execute the cells. It automatically handles the custom compilation of the `chfem` fork and executes the experiment suite. *(Note: PuMA is disabled in the Colab lightweight environment).*

---

## 🧪 Usage

### 1. Running Parameter Sweep Experiments

We provide pre-defined scripts to run specific physical studies. Results (3D `.vti` files, `.png` slices, and logs) are saved into automatically numbered directories (e.g., `result_exp1_01/`), while numerical metrics are aggregated into a central CSV file in the root directory.

```bash
python3 exp/run_exp0_percolation.py     # Standard percolation sweep (length, radius, volume fraction)
python3 exp/run_exp1_agglom.py          # Agglomeration sweep (fiber entanglement and clustering effects)
python3 exp/run_exp2_gyroid.py          # Gyroid background sweep (evaluating double percolation)
python3 exp/run_exp3_hybrid.py          # Hybrid synergy sweep (mixing fibers and flakes)
python3 exp/run_exp4_scale_check.py     # RVE scale, spatial resolution, and computational cost validation
python3 exp/run_exp5_morphology.py      # Phase morphology effects (Lamellar, Cylinder, Gyroid, Sea-Island)
python3 exp/run_exp6_filler.py          # Filler shape reinforcement comparison (Sphere, Flake, Fiber)
python3 exp/run_exp7_shape_stretch.py   # Microstructure deformation under mechanical stretching

# Benchmark & Verification Experiments (chfem vs PuMA)
python3 exp/run_exp8_compare_spheres.py      # Solver comparison for spheres across varying interface profiles
python3 exp/run_exp9_compare_fibers.py       # Solver comparison for isotropic fibers across interface profiles
python3 exp/run_exp10_compare_orientation.py # Pseudo-orientation benchmark evaluating solver accuracy under mechanical stretch
```

> **Note on Benchmarks (Exp 8-10):** These scripts automatically generate `PASS/FAIL` markdown reports and detailed CSV summaries. They verify the consistency between `chfem` and `PuMA` using configured log10-difference thresholds and anisotropy checks.

### 2. Plotting Results

Once the experiments are complete and the central CSV logs are generated, you can visualize the trends and extract structural insights using the included plotting scripts:

```bash
python3 exp/plot_exp0_percolation.py
python3 exp/plot_exp1_agglom.py
python3 exp/plot_exp2_gyroid.py
python3 exp/plot_exp3_hybrid.py
python3 exp/plot_exp4_scale_check.py
python3 exp/plot_exp5_morphology.py
python3 exp/plot_exp6_filler.py
python3 exp/plot_exp7_shape_stretch.py
python3 exp/plot_exp8_compare_spheres.py
python3 exp/plot_exp9_compare_fibers.py
python3 exp/plot_exp10_compare_orientation.py
```

**What these scripts generate:**

1. **Statistical Graphs**: Reads the generated CSVs and outputs comparison charts (e.g., Effective Conductivity vs. Volume Fraction) to validate physical property metrics.
2. **Visual Montages**: Automatically assembles aligned grid images of 2D slices extracted from the 3D microstructures. These montages are organized by key variables (e.g., volume fraction, filler type, or random seed) and maintain physical scaling where appropriate, allowing for an intuitive visual validation of the simulated geometries directly alongside the numerical results.

### 3. Running Custom Pipelines

You can easily design and run a custom simulation by passing arguments directly to `run_pipeline.py`. Use `python3 run_pipeline.py --help` to see all available parameters and recipe formatting.

Recipe strings follow the format:

```text
type:volume_fraction:param1=value1:param2=value2:...
```

Orientation control via `mean_dir` and `kappa` is supported for `flake`, `rigidfiber`, and `irregfiber`.

* `mean_dir=ax,ay,az` sets the preferred orientation axis.
* `kappa=0` gives nearly random orientation.
* Larger `kappa` values produce stronger alignment around `mean_dir`.

The irregular extruded fiber type is `irregfiber` and accepts:

* `length`
* `radius`
* `shape` (`ellipse`, `bean`, or `c-shape`)
* `ratio`
* optional orientation controls: `mean_dir`, `kappa`

```bash
python3 run_pipeline.py \
  --size 200 \
  --bg_type single \
  --physics_mode electrical \
  --solver both \
  --recipe "rigidfiber:0.05:length=60:radius=2" "flake:0.02:radius=15:thickness=2" \
  --basename custom_model \
  --csv_log custom_results.csv
```

**Example: Rigid fibers with preferred alignment along the X direction**

```bash
python3 run_pipeline.py \
  --size 200 \
  --bg_type single \
  --physics_mode thermal \
  --solver skip \
  --recipe "rigidfiber:0.04:length=60:radius=2:mean_dir=1,0,0:kappa=20" \
  --basename aligned_rigidfiber \
  --csv_log aligned_rigidfiber.csv
```

**Example: Flakes with a preferred normal direction**

```bash
python3 run_pipeline.py \
  --size 200 \
  --bg_type single \
  --physics_mode thermal \
  --solver skip \
  --recipe "flake:0.03:radius=14:thickness=2:mean_dir=0,0,1:kappa=30" \
  --basename aligned_flake \
  --csv_log aligned_flake.csv
```

**Example: Irregular extruded fibers with a bean-shaped cross-section**

```bash
python3 run_pipeline.py \
  --size 200 \
  --bg_type single \
  --physics_mode thermal \
  --solver skip \
  --recipe "irregfiber:0.04:length=70:radius=5:shape=bean:ratio=0.55:mean_dir=0,1,0:kappa=25" \
  --basename irregfiber_bean \
  --csv_log irregfiber_bean.csv
```

For `irregfiber`, `radius` is interpreted as the outer or major cross-section scale, while `ratio` controls the secondary geometric scale depending on `shape`.

### 4. Recalculation Mode

You can re-run the computational solvers (chfem, PuMA) on already generated microstructures without undergoing the computationally expensive geometry generation phase. This is highly useful for testing different material properties, running a different solver, or recovering from previous solver errors.

The same pass also recomputes the CSV structure descriptors (`Contact_Ratio`, `Tunneling_Ratio`, `Connectivity_Ratio`, and related voxel counts) from the reconstructed `final_grid` loaded from each `.raw` file.

To use this mode, provide the `--recalc` flag. The script will read the specified `--csv_log`, locate the corresponding `.raw` and `.nf` files in the directory, re-run the solvers, and update the CSV in place.

> **Note:** A backup of your CSV (`[your_csv_name].csv.backup`) is automatically created before any recalculation begins.

**Basic Recalculation:**

```bash
# Re-run both solvers using the existing properties found in the .nf files
python3 run_pipeline.py --recalc --csv_log comparison_results.csv --solver both
```

**Recalculation with Property Overwrite:**
If you want to test new physical properties (e.g., thermal conductivity) on the exact same geometries, use the `--overwrite_props` flag. This forces the script to ignore the old `.nf` properties and apply the new ones provided via the CLI.

```bash
# Overwrite specific properties and re-run only the PuMA solver
python3 run_pipeline.py --recalc --csv_log comparison_results.csv --overwrite_props --prop_A "0.5" --prop_inter "50.0" --solver puma
```

**Key Arguments for Recalculation:**

* `--recalc`: Activates recalculation mode. *(Note: This is mutually exclusive with `--recipe`. You cannot build new models and recalculate old ones in the same command).*
* `--overwrite_props`: Overwrites the material properties in the existing `.nf` files with the new ones provided via CLI arguments.
* `--csv_log`: The target CSV file containing the list of models to process.

---

### 5. Rendering a Review Dashboard from the CSV Log

For quick result review, you can render a self-contained HTML dashboard from a CSV log. The dashboard uses fixed styling, inline data bars, and client-side column sorting implemented with embedded JavaScript. No browser automation or external CDN assets are required.

```bash
python3 render_results_dashboard.py \
  --csv comparison_results.csv \
  --output comparison_results_dashboard.html \
  --sort-by Connectivity_Ratio \
  --descending
```

**What the dashboard provides:**

* A compact review table for the latest CSV results.
* Inline data bars for ratio, fraction, conductivity-like, and count columns.
* Click-to-sort column headers for quick manual inspection in a browser.
* A fully self-contained `.html` output that can be opened offline.

**Useful arguments:**

* `--csv`: Input CSV file path.
* `--output`: Output HTML file path.
* `--columns`: Optional explicit column list and order.
* `--sort-by`: Optional initial sort column.
* `--descending`: Use descending order for the initial sort.
* `--max-rows`: Maximum number of rows to include in the HTML table.
* `--title`, `--subtitle`: Optional labels for the dashboard header.

---

## 📁 File Structure Overview

* `micro_builder.py`: Core logic for 3D microstructure generation, kinematics, and Numba-accelerated RSA (Random Sequential Adsorption) placement.
* `run_pipeline.py`: The main CLI engine bridging structure generation, deformation, and solver execution.
* `exp/run_exp*.py`: Automation scripts for batch experiments.
* `exp/plot_exp*.py`: Matplotlib scripts for generating charts from experiment CSV logs.
* `Dockerfile` / `microsimflow_on_colab.ipynb`: Environment configuration files.

## CSV Output Reference

The central CSV log is intended to be both machine-readable and easy to audit manually.
Existing logs are upgraded in place when new structure-metric columns are added, and `--recalc`
recomputes those fields directly from the saved voxel model.

### Core metadata columns

* `Basename`: Prefix shared by `.raw`, `.nf`, `.vti`, and solver log files.
* `Grid_Size`: Grid dimensions stored as `Nx x Ny x Nz`.
* `Voxel_Size_m`: Physical edge length of one voxel in meters.
* `BG_Type`: Background morphology recipe.
* `Mode`: Physics mode used for the run (`thermal`, `electrical`, or `mechanics`).
* `Solver`: Solver selection used for the run.
* `Recipe`: Full filler recipe string passed from the CLI.
* `Stretch_Ratio`, `Poisson_Ratio`: Deformation parameters for the current output state.

### Phase-fraction columns

* `PolymerA_Frac`, `PolymerB_Frac`: Volume fractions of polymer phases 0 and 1.
* `Secondary_Inter_Frac`: Volume fraction of the secondary interface phase.
* `Primary_Inter_Frac`: Volume fraction of the primary interface phase.
* `Filler_Frac`: Total volume fraction of all filler IDs (`>= 4`).

### Structure descriptor columns

These values are computed as a lightweight post-process from the final voxel grid. They are
meant to track conductive-network trends, not to reconstruct the exact RSA placement history.

* `Contact_Ratio`: `primary interface voxels / filler voxels`
* `Tunneling_Ratio`: `secondary interface voxels / filler voxels`
* `Connectivity_Ratio`: `largest 6-neighbor conductive cluster / all conductive candidate voxels`
* `N_Contact_Voxels`: Number of primary interface voxels.
* `N_Tunnel_Voxels`: Number of secondary interface voxels.
* `N_Filler_Voxels`: Number of filler voxels across all filler IDs.
* `N_Conductive_Candidate_Voxels`: Number of voxels in the conductive mask (`filler + primary + secondary`).
* `N_Largest_Cluster_Voxels`: Size of the largest 6-neighbor connected conductive cluster.
* `N_Conductive_Clusters`: Number of connected components in the conductive mask.

### Solver result columns

* `chfem_Time_s`, `puma_Time_s`: Solver wall time recorded by each backend.
* `chfem_Txx` ... `chfem_Txy`: Homogenized tensor components from chfem.
* `puma_Txx` ... `puma_Txy`: Homogenized tensor components from PuMA.
  PuMA currently fills only the diagonal terms in this wrapper.

### Optional control

* `--skip_structure_metrics`: Skip the lightweight structure-descriptor calculation.
  The default behavior is to compute and export them for every run and every recalculation.
