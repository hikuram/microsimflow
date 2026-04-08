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
python3 run_exp0_percolation.py   # Standard percolation sweep (length, radius, volume fraction)
python3 run_exp1_agglom.py        # Agglomeration sweep (fiber entanglement effects)
python3 run_exp2_gyroid.py        # Gyroid background sweep
python3 run_exp3_hybrid.py        # Hybrid synergy sweep (mixing fibers and flakes)
python3 run_exp4_scale_check.py   # Representative Volume Element (RVE) scale validation
```

### 2. Plotting Results
Once the experiments are complete and the central CSV logs are generated, you can visualize the trends using the included plotting scripts:

```bash
python3 plot_exp0_percolation.py
python3 plot_exp1_agglom.py
python3 plot_exp2_gyroid.py
python3 plot_exp3_hybrid.py
```
*These scripts will read the generated CSVs and output comparison graphs.*

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

---

## 📁 File Structure Overview
* `micro_builder.py`: Core logic for 3D microstructure generation and RSA (Random Sequential Adsorption) placement.
* `run_pipeline.py`: The main CLI engine bridging structure generation and solver execution.
* `run_exp*.py`: Automation scripts for batch experiments.
* `plot_exp*.py`: Matplotlib scripts for generating charts from experiment CSV logs.
* `Dockerfile` / `microsimflow_on_colab.ipynb`: Environment configuration files.
