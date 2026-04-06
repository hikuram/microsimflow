# microsimflow

A Python package integrating custom microstructure modeling with property evaluations (thermal, electrical, and mechanical) using the `chfem` and `PuMA` solvers.

This repository is designed to provide a flat script structure from structure generation to solver execution and result visualization, allowing you to reproduce and expand experiments without managing complex directory hierarchies.

## Key Features
* **Diverse Background Phases**: Single phase, Gyroid (co-continuous), Lamellar, Cylinder, BCC, Sea-Island structures, etc.
* **Adaptive Filler Placement**: Rigid spheres, flakes, and rigid cylinders, as well as the generation of flexible fibers and agglomerates that grow adaptively to the topology.
* **Dual Solver Integration**: 
  * `chfem`: High-efficiency homogenization solver with GPU support.
  * `PuMA`: Multipurpose solver for the Laplace equation.

## File Structure
* `micro_builder.py`: Core module for microstructure generation logic.
* `run_pipeline.py`: Main CLI tool bridging structure generation and solver execution.
* `run_exp*.py`: Automation scripts for running various parameter sweep experiments.
* `plot_exp*.py`: Scripts for generating plots (PNG) from experiment results (CSV).
* `Dockerfile`: Container environment based on `nvidia/cuda:12.9.1-devel-ubuntu22.04` with required Python libraries and pre-compiled `chfem`.

## Environment Setup
It is recommended to build the environment using the provided `Dockerfile`.

```bash
# Build the Docker image
docker build -t microsim_env .

# Run the container (Jupyter Lab starts on port 8888)
docker run -it --rm --gpus all -v $(pwd):/workspace microsim_env bash
```

## Usage
### Running Experiment Scripts
Run parameter sweep experiments with pre-defined conditions.

```bash
python3 run_exp1_agglom.py
python3 run_exp2_gyroid.py
python3 run_exp3_hybrid.py
```
Executing each script will output structure data (`.vti`) and slice images (`.png`) in the `result_exp*/` directories, and generate a `.csv` summarizing the results in the root directory.

### Plotting Results
After the experiments are complete, you can generate graphs with the following scripts:

```bash
python3 plot_exp1_agglom.py
python3 plot_exp2_gyroid.py
python3 plot_exp3_hybrid.py
```

### Running Custom Pipelines
You can run simulations under custom conditions by passing arguments to `run_pipeline.py`.

```bash
python3 run_pipeline.py --size 200 --bg_type single --physics_mode electrical --solver chfem --recipe "rigidfiber:0.05:length=60:radius=2" --basename custom_model
```
