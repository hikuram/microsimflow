# microsimflow

<p align="center">
  <img src="https://raw.githubusercontent.com/hikuram/microsimflow/main/docs/logo.png" alt="microsimflow Logo" width="600">
</p>

A Python-based workflow integrating custom 3D microstructure modeling with property evaluations (thermal, electrical, and mechanical) using the `chfem` and `PuMA` solvers.

This repository provides a modular, end-to-end simulation framework encompassing 3D structure generation, solver execution, and result visualization. It features robust experiment management, allowing you to seamlessly reproduce and expand large-scale parameter sweeps without data loss.

Each run writes a CSV log that stores both solver outputs and lightweight structure descriptors computed directly from the generated `final_grid`. These descriptors are also available in `--recalc` mode, so legacy models can be re-evaluated without rebuilding geometry.

## ✨ Key Features
* **Diverse Background Phases**: Single phase, Gyroid (co-continuous), Lamellar, Cylinder, BCC, Sea-Island structures.
* **Adaptive Filler Placement**: Rigid spheres, flakes, rigid cylinders, irregular-cross-section extruded fibers (`irregfiber`), and topology-adaptive flexible fibers/agglomerates.
* **Orientation-Controlled Fillers**: Supports directional alignment using a unified **Pseudo-Watson distribution**. Control orientation via `mean_dir=ax,ay,az` and `kappa`.
  * `kappa = 0`: Isotropic random orientation.
  * `kappa > 0`: Polar alignment (vMF style) strictly along the `mean_dir` axis.
  * `kappa < 0`: Girdle/Equatorial alignment where vectors distribute in the plane perpendicular to the `mean_dir` axis.
* **Kinematic Deformation**: Simulate affine mechanical stretching with dynamic rigid-body kinematics for rigid fillers, outputting robust VTM/PVD time-series collections for ParaView.
* **Dual Solver Integration**: 
  * `chfem`: High-efficiency homogenization solver with GPU support. *(Note: This environment compiles a custom fork to enable the specific build options required for models with 5 or more properties).*
  * `PuMA`: Multipurpose solver for the Laplace equation.
* **Robust Experiment Management**: Automated sequential directory creation (`result_expX_01/`, `02/`...) prevents data overwriting, while all metrics are safely appended to a central `.csv` log with real-time disk syncing.
* **Cloud & Local Ready**: Includes a `Dockerfile` for local GPU workstations and a Jupyter Notebook for instant execution on Google Colab (T4 GPU).

## 📚 Documentation
For detailed usage, recipes, and architecture, please refer to the `docs/` directory:

1. **[CLI Usage & Custom Pipelines](docs/USAGE_CLI.md)**: How to run custom models, define filler recipes (including orientation), and use Recalculation mode.
2. **[Batch Experiments & Plotting](docs/EXPERIMENTS_CATALOG.md)**: Overview of pre-defined parameter sweep scripts and visualization tools.
3. **[Output Data & CSV Reference](docs/OUTPUT_DATA_FORMAT.md)**: Explanation of the generated files, descriptors, metrics, and CSV log structure.
4. **[Algorithm Architecture](docs/RSA_ALGORITHM_ARCHITECTURE.md)**: Deep dive into the Kinematic RSA algorithm, Numba optimizations, and physics-aware interface generation.

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
```

### Option B: Google Colab (Cloud)

If you do not have a local GPU workstation, you can use the provided `microsimflow_on_colab.ipynb`.
Upload the notebook to Google Colab, set the runtime to **T4 GPU**, and execute the cells. It automatically handles the custom compilation of the `chfem` fork and executes the experiment suite. *(Note: PuMA is disabled in the Colab lightweight environment).*

---

## 📁 File Structure Overview

* `micro_builder.py`: Core logic for 3D microstructure generation, kinematics, and Numba-accelerated RSA (Random Sequential Adsorption) placement.
* `run_pipeline.py`: The main CLI engine bridging structure generation, deformation, and solver execution.
* `exp/run_exp*.py`: Automation scripts for batch experiments.
* `exp/plot_exp*.py`: Matplotlib scripts for generating charts from experiment CSV logs.
* `Dockerfile` / `microsimflow_on_colab.ipynb`: Environment configuration files.
