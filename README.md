# microsimflow

A Python package integrating custom 3D microstructure modeling with property evaluations (thermal, electrical, and mechanical) using the `chfem` and `PuMA` solvers.

This repository provides an end-to-end flat script structure—from structure generation to solver execution and result visualization. It features robust experiment management, allowing you to reproduce and expand large-scale parameter sweeps without managing complex directory hierarchies or worrying about data loss.

## ✨ Key Features
* **Diverse Background Phases**: Single phase, Gyroid (co-continuous), Lamellar, Cylinder, BCC, Sea-Island structures.
* **Adaptive Filler Placement**: Rigid spheres, flakes, rigid cylinders, and topology-adaptive flexible fibers/agglomerates.
* **Dual Solver Integration**: 
  * `chfem`: High-efficiency homogenization solver with GPU support. *(Note: This environment compiles a custom fork to enable the specific build options required for models with 5 or more properties).*
  * `PuMA`: Multipurpose solver for the Laplace equation.
* **Robust Experiment Management**: Automated sequential directory creation (`result_expX_01/`, `02/`...) prevents data overwriting, while all metrics are safely appended to a central `.csv` log.
* **Cloud & Local Ready**: Includes a `Dockerfile` for local GPU workstations and a Jupyter Notebook for instant execution on Google Colab (T4 GPU).

---

## 🚀 Environment Setup

You can run `microsimflow` either locally using Docker or in the cloud using Google Colab.

### Option A: Local Run via Docker (Recommended)
It is recommended to build the environment using the provided `Dockerfile`. 
This process installs Python dependencies, `PuMA`, and **compiles `chfem` from source** (CUDA 12.9). 

*Important: Because the official `chfem` repository currently fails to build when passing the `-DCUDAPCG_MATKEY_32BIT` flag (required for models with 5+ properties), the Dockerfile uses a temporarily forked version with the necessary typo fixes.*

```bash
# Build the Docker image
docker build -t microsim_env .

# Run the container
docker run -it --rm --gpus all -v $(pwd):/workspace microsim_env bash
```

### Option B: Google Colab (Cloud)
If you do not have a local GPU, you can use the provided `microsimflow_on_colab.ipynb`. 
Just upload the notebook to Google Colab, set the runtime to **T4 GPU**, and run the cells. It automatically compiles a compatible version of `chfem` and executes the experiment suite. *(Note: PuMA is disabled in the Colab lightweight environment).*

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
```

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
```

**What these scripts generate:**
1. **Statistical Graphs**: Reads the generated CSVs and outputs comparison charts (e.g., Effective Conductivity vs. Volume Fraction) to validate physical property metrics.
2. **Visual Montages**: Automatically assembles aligned grid images of 2D slices extracted from the 3D microstructures. These montages are organized by key variables (e.g., volume fraction, filler type, or random seed) and maintain physical scaling where appropriate, allowing for an intuitive visual validation of the simulated geometries directly alongside the numerical results.

### 3. Running Custom Pipelines
You can easily design and run a custom simulation by passing arguments directly to `run_pipeline.py`.

```bash
python3 run_pipeline.py \
  --size 200 \
  --bg_type single \
  --physics_mode electrical \
  --solver both \
  --recipe "rigidfiber:0.05:length=60:radius=2" \
  --basename custom_model \
  --csv_log custom_results.csv
```

### 4. Recalculation Mode

You can re-run the computational solvers (chfem, PuMA) on already generated microstructures without undergoing the computationally expensive geometry generation phase. This is highly useful for testing different material properties, running a different solver, or recovering from previous solver errors.

To use this mode, provide the `--recalc` flag. The script will read the specified `--csv_log`, locate the corresponding `.raw` and `.nf` files in the directory, re-run the solvers, and update the CSV in place.

> **Note:** A backup of your CSV (`[your_csv_name].csv.backup`) is automatically created before any recalculation begins.

**Basic Recalculation:**
```bash
# Re-run both solvers using the existing properties found in the .nf files
python run_pipeline.py --recalc --csv_log comparison_results.csv --solver both
```

**Recalculation with Property Overwrite:**
If you want to test new physical properties (e.g., thermal conductivity) on the exact same geometries, use the `--overwrite_props` flag. This forces the script to ignore the old `.nf` properties and apply the new ones provided via the CLI.

```bash
# Overwrite specific properties and re-run only the PuMA solver
python run_pipeline.py --recalc --csv_log comparison_results.csv --overwrite_props --prop_A "0.5" --prop_inter "50.0" --solver puma
```

**Key Arguments for Recalculation:**
* `--recalc`: Activates recalculation mode. (Note: This is mutually exclusive with `--recipe`. You cannot build new models and recalculate old ones in the same command).
* `--overwrite_props`: Overwrites the material properties in the existing `.nf` files with the new ones provided via CLI arguments.
* `--csv_log`: The target CSV file containing the list of models to process.

---

## 📁 File Structure Overview
* `micro_builder.py`: Core logic for 3D microstructure generation and RSA (Random Sequential Adsorption) placement.
* `run_pipeline.py`: The main CLI engine bridging structure generation and solver execution.
* `exp/run_exp*.py`: Automation scripts for batch experiments.
* `exp/plot_exp*.py`: Matplotlib scripts for generating charts from experiment CSV logs.
* `Dockerfile` / `microsimflow_on_colab.ipynb`: Environment configuration files.
