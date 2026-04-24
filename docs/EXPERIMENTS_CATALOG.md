# Batch Experiments & Plotting

We provide pre-defined scripts to run specific physical studies. Results are saved into automatically numbered directories, while metrics are aggregated into a central CSV file.

## Running Parameter Sweep Experiments

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

## Plotting Results

Once experiments are complete, visualize trends and extract structural insights using plotting scripts:

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
1. **Statistical Graphs**: Comparison charts (e.g., Effective Conductivity vs. Volume Fraction).
2. **Visual Montages**: Automatically assembled grid images of 2D slices extracted from 3D microstructures, organized by key variables.
