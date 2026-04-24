# Output Data & CSV Reference

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
