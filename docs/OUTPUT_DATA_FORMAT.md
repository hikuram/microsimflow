# Output Data & CSV Reference

## VTI Visualization Output

Along with the CSV log, `microsimflow` generates a 3D visualization file in VTI format (`[basename].vti`) and a corresponding VTM wrapper for time-series/stretch evaluation in ParaView.

By default, if the computational solvers are executed, the pipeline will automatically extract the resulting physical fields (e.g., pressure, temperature, stress) and embed them directly into the VTI file's `cell_data`. This allows you to seamlessly visualize both the microstructure's phase geometry and its physical response simultaneously.

> **Optional Control (`--vti_fields`):**
> Embedding these fields can significantly increase the `.vti` file size. You can control this behavior using the `--vti_fields [on/off]` argument. Setting it to `off` will only export the structural `Phase` ID.

### Available Physical Fields in VTI

Depending on the selected `--physics_mode`, the following fields are automatically mapped and embedded from the solver's binary outputs:

| Physics Mode | VTI Field Name | Type | chfem Output Binary | Description |
| :--- | :--- | :--- | :--- | :--- |
| **All Modes** | `Phase` | Scalar | *(Generated Internally)* | Voxel-based structural phase ID |
| **Permeability** | `Pressure` | Scalar | `[basename]_pressure_0.bin` | Fluid pressure distribution |
| | `Velocity` | Vector | `[basename]_velocity_0.bin` | Fluid velocity vector (Vx, Vy, Vz) |
| **Thermal** | `Temperature` | Scalar | `[basename]_temperature_0.bin` | Temperature distribution |
| | `Heat_Flux` | Vector | `[basename]_heat_flux_0.bin` | Heat flux vector |
| **Electrical** | `Potential` | Scalar | `[basename]_potential_0.bin` | Electric potential distribution |
| | `Electric_Current` | Vector | `[basename]_electric_current_0.bin` | Electric current density vector |
| **Mechanics** | `Displacement` | Vector | `[basename]_displacement_0.bin` | Displacement vector |
| | `Stress` | Tensor | `[basename]_stress_0.bin` | Stress tensor (6 components: xx, yy, zz, yz, zx, xy) |

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

### Taufactor descriptor columns (Structural Tortuosity)

These structural descriptors are calculated using the `taufactor` module and fully account for Periodic Boundary Conditions (PBC). They serve as low-cost, high-throughput structural screening indicators prior to running full homogenization solvers.

* `tau_X`, `tau_Y`, `tau_Z`: Tortuosity factor along each respective axis.
* `D_eff_X`, `D_eff_Y`, `D_eff_Z`: Effective diffusivity along each respective axis.
* `tau_Time_s`: Execution time required for the `taufactor` calculation in seconds.

> **Note:** In `permeability` mode, the grid is automatically binarized before calculation: specified `void_phases` are treated as fluid (1) and all other phases as solid (0). In all other modes, the multi-phase conductivity map defined by the `prop_map` is used directly.

### Solver result columns

* `chfem_Time_s`, `puma_Time_s`: Solver wall time recorded by each backend.
* `chfem_Txx` ... `chfem_Txy`: Homogenized tensor components from chfem.
* `puma_Txx` ... `puma_Txy`: Homogenized tensor components from PuMA.
  PuMA currently fills only the diagonal terms in this wrapper.

### Optional control

* `--skip_structure_metrics`: Skip the lightweight structure-descriptor calculation.
  The default behavior is to compute and export them for every run and every recalculation.
