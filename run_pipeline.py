import argparse
import time
import csv
import os
import re
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Import from micro_builder
from micro_builder import (
    set_random_seed,
    build_single_phase_grid,
    build_tpms_grid_with_target_ratio,
    build_sea_island_grid,
    build_island_sea_grid,
    build_lamellar_grid,
    build_cylinder_hex_grid,
    build_bcc_grid,
    place_fillers_hybrid,
    place_adaptive_fibers,
    get_flake_mask,
    get_sphere_mask,
    get_rigid_cylinder_mask,
    get_flexible_fiber_mask,
    get_agglomerate_mask,
    get_staggered_flakes_mask,
    finalize_microstructure,
    export_chfem_inputs,
    export_visualization_vti,
    summarize_phase_fractions,
    compute_structure_metrics,
    apply_background_deformation,
    render_deformed_fillers
)

def parse_chfem_log(log_path):
    """Extract up to 6 diagonal tensor components (General Txx-Txy notation) and computation time"""
    # Initialize with empty strings for all 6 possible components (xx, yy, zz, yz, zx, xy)
    diag = [""] * 6
    total_time = 0.0
    try:
        if not os.path.exists(log_path):
            return diag, 0.0
            
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract the homogenized constitutive matrix block
        mat_match = re.search(r'Homogenized Constitutive Matrix.*?:[^\n]*\n(.*?)\n-{10,}', content, re.DOTALL)
        if mat_match:
            lines = [line.strip() for line in mat_match.group(1).strip().split('\n') if line.strip()]
            # Extract diagonal elements dynamically based on matrix size (3x3 or 6x6)
            for i in range(min(len(lines), 6)):
                parts = lines[i].split()
                if len(parts) > i:
                    diag[i] = float(parts[i])

        # Sum up elapsed time matches
        time_matches = re.findall(r'Elapsed time(?: \(total\))?:\s*([\d\.eE\+\-]+)', content)
        if time_matches:
            total_time = sum(float(t) for t in time_matches)
    except Exception as e:
        print(f"chfem log parsing error: {e}")

    return diag, total_time

def parse_nf_properties(nf_path):
    """Read and parse property map from an existing .nf file"""
    prop_map = {}
    if not os.path.exists(nf_path):
        return prop_map
        
    with open(nf_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_props = False
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('%properties_of_materials'):
            in_props = True
            continue
        elif line.startswith('%') and in_props:
            in_props = False # Reached another section
            
        if in_props:
            parts = line.split()
            if len(parts) >= 2:
                phase_id = int(parts[0])
                prop_val = " ".join(parts[1:])
                prop_map[phase_id] = prop_val
                
    return prop_map


STRUCTURE_METRIC_COLUMNS = [
    "Contact_Ratio", "Tunneling_Ratio", "Connectivity_Ratio",
    "N_Contact_Voxels", "N_Tunnel_Voxels", "N_Filler_Voxels",
    "N_Conductive_Candidate_Voxels", "N_Largest_Cluster_Voxels", "N_Conductive_Clusters"
]


def structure_metrics_to_csv_fields(metrics):
    """Format structure metrics for stable CSV output."""
    return {
        "Contact_Ratio": f"{metrics['contact_ratio']:.4f}",
        "Tunneling_Ratio": f"{metrics['tunneling_ratio']:.4f}",
        "Connectivity_Ratio": f"{metrics['connectivity_ratio']:.4f}",
        "N_Contact_Voxels": str(metrics['n_contact_voxels']),
        "N_Tunnel_Voxels": str(metrics['n_tunnel_voxels']),
        "N_Filler_Voxels": str(metrics['n_filler_voxels']),
        "N_Conductive_Candidate_Voxels": str(metrics['n_conductive_candidate_voxels']),
        "N_Largest_Cluster_Voxels": str(metrics['n_largest_cluster_voxels']),
        "N_Conductive_Clusters": str(metrics['n_conductive_clusters']),
    }


def ensure_structure_metric_columns(header, rows):
    """Append missing structure-metric columns and pad existing rows."""
    missing = [col for col in STRUCTURE_METRIC_COLUMNS if col not in header]
    if not missing:
        return header, rows

    header = header + missing
    for row in rows:
        row.extend([""] * len(missing))
    return header, rows


def upgrade_existing_csv_log(csv_path):
    """Upgrade an existing CSV log in place when new structure-metric columns are missing."""
    if not os.path.exists(csv_path):
        return

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        rows = list(reader)

    new_header, new_rows = ensure_structure_metric_columns(header, rows)
    if new_header == header:
        return

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

def run_recalculation_mode(args):
    """Executes the recalculation pipeline using existing .raw and .nf files"""
    if not os.path.exists(args.csv_log):
        print(f"Error: CSV file '{args.csv_log}' not found for recalculation.")
        return

    # 1. Create a backup of the original CSV
    backup_path = args.csv_log + ".backup"
    shutil.copy2(args.csv_log, backup_path)
    print(f"--- Recalculation Mode Started ---")
    print(f"Backup created at: {backup_path}")

    # Fallback properties from command-line arguments (used if .nf is missing or new IDs found)
    if args.physics_mode == 'thermal':
        fallback_props = {0: "0.3", 1: "0.3", 2: "30.0", 3: "30.0", 4: "300.0"}
    elif args.physics_mode == 'electrical':
        fallback_props = {0: "1e-4", 1: "1e-4", 2: "1e-3", 3: "1e-1", 4: "1e4"}
    else:  # mechanics
        fallback_props = {0: "3.0 1.0", 1: "3.0 1.0", 2: "10.0 3.0", 3: "15.0 5.0", 4: "100.0 50.0"}

    # Load the original CSV into memory.
    # The recalculation loop updates only refreshed values, then rewrites the full CSV after each row.
    with open(args.csv_log, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    header, rows = ensure_structure_metric_columns(header, rows)

    # Required metadata columns for identifying files and grid dimensions
    try:
        idx_basename = header.index("Basename")
        idx_grid_size = header.index("Grid_Size")
        idx_voxel_size = header.index("Voxel_Size_m")
        idx_mode = header.index("Mode")
    except ValueError as e:
        print(f"Error: CSV is missing required columns ({e})")
        return

    # 2. Iterate through each model in the CSV
    for row_idx, row in enumerate(rows):
        basename = row[idx_basename]
        grid_size_str = row[idx_grid_size]
        voxel_size = float(row[idx_voxel_size])
        mode = row[idx_mode]

        print(f"\nRecalculating: {basename}")

        raw_file = f"{basename}.raw"
        nf_file = f"{basename}.nf"

        if not os.path.exists(raw_file):
            print(f"  Skipping: {raw_file} not found.")
            continue

        # Restore grid to determine unique Phase IDs
        dim_parts = grid_size_str.split('x')
        nx, ny, nz = int(dim_parts[0]), int(dim_parts[1]), int(dim_parts[2])
        final_grid = np.fromfile(raw_file, dtype=np.uint8).reshape((nz, ny, nx))

        # 3. Resolve Property Map (Task 5 Fix: Preserve heterogeneity)
        prop_map = parse_nf_properties(nf_file)

        if args.overwrite_props or not prop_map:
            print("  Updating properties from command-line arguments...")
            cli_overrides = {0: args.prop_A, 1: args.prop_B, 2: args.prop_inter2, 3: args.prop_inter}
            unique_ids = np.unique(final_grid)

            for uid in unique_ids:
                # Priority 1: User-specified CLI override
                if uid in cli_overrides and cli_overrides[uid] is not None:
                    prop_map[uid] = cli_overrides[uid]
                # Priority 2: Keep existing .nf property if present (protects multiple filler IDs)
                elif uid in prop_map:
                    continue
                # Priority 3: Hard fallback for missing data
                else:
                    if uid in fallback_props:
                        prop_map[uid] = fallback_props[uid]
                    elif uid >= 4:
                        prop_map[uid] = fallback_props[4]

            # Re-export .nf for chfem with corrected properties
            export_chfem_inputs(final_grid, basename, voxel_size, mode, prop_map)

        # 4. Compute lightweight structure metrics directly from final_grid
        if args.skip_structure_metrics:
            structure_metrics = {
                'contact_ratio': 0.0, 'tunneling_ratio': 0.0, 'connectivity_ratio': 0.0,
                'n_contact_voxels': 0, 'n_tunnel_voxels': 0, 'n_filler_voxels': 0,
                'n_conductive_candidate_voxels': 0, 'n_largest_cluster_voxels': 0, 'n_conductive_clusters': 0,
            }
        else:
            structure_metrics = compute_structure_metrics(final_grid)
        metric_fields = structure_metrics_to_csv_fields(structure_metrics)

        # 5. Execute solvers and collect results (Task 4: Universal T-notation)
        chfem_time = ""
        chfem_results = [""] * 6
        puma_time = ""
        puma_results = [""] * 6

        if args.solver in ["chfem", "both"]:
            log_file = f"{basename}_metrics.txt"
            subprocess.run(["chfem_exec", nf_file, raw_file, "-m", log_file])
            res_diag, ctime = parse_chfem_log(log_file)
            if res_diag[0] != "":
                chfem_time, chfem_results = f"{ctime:.2f}", res_diag

        if args.solver in ["puma", "both"]:
            cond_map = {k: float(v.split()[0]) for k, v in prop_map.items()}
            pkx, pky, pkz, ptime = run_puma_laplace(final_grid, voxel_size, mode, cond_map)
            if pkx is not None:
                puma_time = f"{ptime:.2f}"
                puma_results = [pkx, pky, pkz, "", "", ""]

        # 6. Update the row with new metrics
        try:
            row[header.index("chfem_Time_s")] = chfem_time
            row[header.index("puma_Time_s")] = puma_time

            # Map results to universal Txx, Tyy, Tzz, Tyz, Tzx, Txy columns
            comp_suffixes = ["xx", "yy", "zz", "yz", "zx", "xy"]
            for i, suffix in enumerate(comp_suffixes):
                col_c = f"chfem_T{suffix}"
                col_p = f"puma_T{suffix}"
                if col_c in header:
                    row[header.index(col_c)] = chfem_results[i]
                if col_p in header:
                    row[header.index(col_p)] = puma_results[i]

            for col_name, value in metric_fields.items():
                if col_name in header:
                    row[header.index(col_name)] = value
        except ValueError:
            pass

        rows[row_idx] = row

        # Persist the current CSV state after each updated row.
        # This preserves progressive saving without rebuilding from an empty file.
        with open(args.csv_log, mode='w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)
            writer.writerows(rows)
            out_f.flush()
            os.fsync(out_f.fileno())

    print(f"\n--- Recalculation Complete. Saved to {args.csv_log} ---")

def run_puma_laplace(final_grid, voxel_size, physics_mode, cond_map):
    """Solve the Laplace equation (thermal/electrical conduction) using PuMA's Python API"""
    try:
        import pumapy as puma
    except ImportError:
        print("Error: pumapy module not found. Cannot run PuMA solver.")
        return None, None, None, 0.0

    if physics_mode == 'mechanics':
        print("Mechanics mode is currently routed to chfem only in this wrapper. Skipping PuMA.")
        return None, None, None, 0.0

    # Transpose dimensions: NumPy (Z, Y, X) -> PuMA Workspace (X, Y, Z)
    ws = puma.Workspace.from_array(final_grid.transpose(2, 1, 0))
    ws.voxel_length = voxel_size

    puma_cond_map = puma.IsotropicConductivityMap()
    for phase_id, cond_val in cond_map.items():
        puma_cond_map.add_material((int(phase_id), int(phase_id)), float(cond_val))

    print("\n--- Running PuMA Solver ---")
    t0 = time.time()

    try:
        # Compute for each XYZ direction (specify periodic boundary conditions with side_bc='p')
        print("Computing X direction...")
        res_x = puma.compute_thermal_conductivity(ws, puma_cond_map, direction='x', side_bc='p', solver_type='cg')
        txx = res_x[0] if isinstance(res_x, tuple) else res_x
        
        print("Computing Y direction...")
        res_y = puma.compute_thermal_conductivity(ws, puma_cond_map, direction='y', side_bc='p', solver_type='cg')
        tyy = res_y[0] if isinstance(res_y, tuple) else res_y
        
        print("Computing Z direction...")
        res_z = puma.compute_thermal_conductivity(ws, puma_cond_map, direction='z', side_bc='p', solver_type='cg')
        tzz = res_z[0] if isinstance(res_z, tuple) else res_z
        
        total_time = time.time() - t0
        print(f"PuMA computation completed in {total_time:.2f}s")
        
        return txx, tyy, tzz, total_time

    except Exception as e:
        print(f"PuMA encountered an error during computation: {e}")
        return None, None, None, 0.0

def save_thumbnail_png(grid, filename):
    """
    Save the Z-axis center slice as a PNG.
    Capping values at 4 ensures that Filler 2 (ID 5) and beyond 
    are colored identically to Filler 1 (ID 4).
    """
    num_phases = 5
    vmin, vmax = -0.5, 4.5
    custom_cmap = plt.get_cmap('viridis', num_phases)

    z_mid = grid.shape[0] // 2
    # Clip all filler IDs (4, 5, 6...) to 4 for consistent visualization
    slice_img = np.clip(grid[z_mid, :, :], 0, 4)
    
    # Save the pure 2D array directly to a PNG file without any Matplotlib figure overhead
    plt.imsave(filename, slice_img, cmap=custom_cmap, vmin=vmin, vmax=vmax, origin='upper')
    print(f"Saved clean thumbnail: {filename}")

def export_vtm_wrapper(vti_filepath, vtm_filepath):
    """
    Creates a lightweight VTM (MultiBlock) wrapper that references the original VTI file.
    This safely bypasses ParaView's extent caching issue during volume rendering.
    """
    # Calculate relative path from 'pvd/' directory to the original VTI file (../file.vti)
    rel_vti_path = f"../{os.path.basename(vti_filepath)}"
    
    vtm_content = f"""<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">
  <vtkMultiBlockDataSet>
    <DataSet index="0" file="{rel_vti_path}"/>
  </vtkMultiBlockDataSet>
</VTKFile>
"""
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(vtm_filepath), exist_ok=True)
    
    with open(vtm_filepath, 'w', encoding='utf-8') as f:
        f.write(vtm_content)

def update_pvd_file(pvd_filepath, dataset_records):
    """
    Creates or updates a ParaView Data (.pvd) file to group VTM wrappers as a time-series.
    This ensures ParaView correctly handles varying grid extents during deformation.
    """
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(pvd_filepath), exist_ok=True)
    
    with open(pvd_filepath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for timestep, filepath in dataset_records:
            # Extract basename for the 'file' attribute.
            # Since both PVD and VTM are in the 'pvd/' directory, only the filename is needed.
            rel_filename = os.path.basename(filepath)
            f.write(f'    <DataSet timestep="{timestep}" group="" part="0" file="{rel_filename}"/>\n')
            
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

def export_common_legend(filename="common_legend.png"):
    """
    Exports a standalone legend image with fixed 5 phases.
    Moving this outside the loop ensures consistency across all experiments.
    """
    if os.path.exists(filename):
        return
        
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif'],
        'axes.labelsize': 12, 'ytick.labelsize': 12, 
        'savefig.bbox': 'tight'
    })

    # Standardized 5 phases
    ids = [0, 1, 2, 3, 4]
    labels = ['Polymer A', 'Polymer B', 'Secondary Inter', 'Primary Inter', 'Filler']
    num_phases = 5
    
    vmin, vmax = -0.5, 4.5
    custom_cmap = plt.get_cmap('viridis', num_phases)

    fig, ax = plt.subplots(figsize=(1.5, 4)) 
    ax.axis('off')
    
    # Create a dummy ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, ticks=ids, fraction=1.0, pad=0.0)
    cbar.ax.set_yticklabels(labels)
    cbar.ax.set_title("Phase ID", pad=15, fontweight='bold')

    plt.savefig(filename, dpi=200, transparent=False, facecolor='white')
    plt.close(fig)
    print(f"Saved common legend: {filename}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="microsimflow: A workflow for 3D microstructure generation and property homogenization.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--size", type=int, default=200, help="Grid size for the 3D microstructure (default: 200).")
    parser.add_argument("--voxel_size", type=float, default=1e-8, help="Physical size of one voxel in meters (default: 1e-8).")
    parser.add_argument("--bg_type", type=str, default="gyroid",
                        choices=["single", "gyroid", "sea_island", "island_sea", "lamellar", "cylinder", "bcc"],
                        help="Type of continuous polymer background phase (default: gyroid).")
    parser.add_argument("--phaseA_ratio", type=float, default=0.50, help="Target volume fraction for Phase A in the background (default: 0.50).")

    group = parser.add_mutually_exclusive_group(required=True)
    
    recipe_help = """Recipe for filler placement (required for new build).
Format: type:volume_fraction:param1=val1:param2=val2...
Available types and common parameters:
  - flake        : radius, thickness
  - sphere       : radius
  - rigidfiber   : length, radius
  - adaptfiber   : length, radius, max_bend_deg, max_total_bends, max_protrusion_ratio
  - flexfiber    : length, radius, max_bend_deg, max_total_bends
  - agglomerate  : num_fibers, length, radius, max_bend_deg, max_total_bends
  - staggered    : radius, layer_thickness, min_layers, max_layers, max_offset_pct
Optional param: 'prop=X' to override the physical property for this specific filler.
Example: --recipe "rigidfiber:0.05:length=60:radius=2:prop=500.0" "flake:0.02:radius=15:thickness=2"
"""
    group.add_argument("--recipe", nargs='+', help=recipe_help)
    group.add_argument("--recalc", action="store_true", help="Launch in recalculation mode (skip model generation, use existing .raw/.nf files).")
    
    parser.add_argument("--basename", type=str, default="model", help="Base filename for generated files (default: 'model').")
    parser.add_argument("--csv_log", type=str, default="comparison_results.csv", help="CSV file to append/update results (default: 'comparison_results.csv').")
    parser.add_argument("--physics_mode", type=str, default="thermal",
                        choices=["thermal", "electrical", "mechanics"],
                        help="Physics mode which defines interface handling and default properties (default: thermal).")
    parser.add_argument("--solver", type=str, default="both",
                        choices=["chfem", "puma", "both", "skip"],
                        help="Solver for homogenisation. Use 'skip' to only build the model without running solvers (default: both).")
    parser.add_argument("--prop_A", type=str, default=None, help="Property value for Polymer A (Phase 0).")
    parser.add_argument("--prop_B", type=str, default=None, help="Property value for Polymer B (Phase 1).")
    parser.add_argument("--prop_inter2", type=str, default=None, help="Property value for Secondary Contact/Tunnel phase (Phase 2).")
    parser.add_argument("--prop_inter", type=str, default=None, help="Property value for Primary Contact/Tunnel phase (Phase 3).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--tunnel_radius", type=int, default=2, 
                        help="Radius in voxels for nearest-neighbor (tunneling/secondary) detection (default: 2).")
    parser.add_argument("--contact_radius", type=int, default=1, 
                        help="Radius in voxels for primary contact interface thickness (default: 1).")
    parser.add_argument("--stretch_ratios", type=float, nargs='+', default=[1.0], 
                        help="List of stretch ratios (lambda) along X-axis to evaluate deformation (default: 1.0).")
    parser.add_argument("--poisson_ratio", type=float, default=0.4, 
                        help="Poisson's ratio (nu) for transverse compression during deformation (default: 0.4).")
    parser.add_argument("--overwrite_props", action="store_true", help="Overwrite .nf properties with command-line arguments during recalculation.")
    parser.add_argument("--skip_structure_metrics", action="store_true", help="Skip lightweight post-process structure metrics (enabled by default).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- Recalculation Mode Branch ---
    if getattr(args, 'recalc', False):
        run_recalculation_mode(args)
        return  # Exit here if in recalculation mode
    
    # --- Fix the generator if a seed is specified ---
    if args.seed is not None:
        set_random_seed(args.seed)
        
    # --- Pre-computation Setup ---
    upgrade_existing_csv_log(args.csv_log)

    # Generate the common legend once before entering any loops
    export_common_legend()
    
    # Set default physical properties according to the physics mode
    if args.physics_mode == 'thermal':
        prop_A = args.prop_A or "0.3"
        prop_B = args.prop_B or "0.3"
        prop_secondary_inter = args.prop_inter2 or "30.0" # Safe fallback even if not present in grid
        prop_primary_inter = args.prop_inter or "30.0"
        default_filler = "300.0"
    elif args.physics_mode == 'electrical':
        prop_A = args.prop_A or "1e-4"
        prop_B = args.prop_B or "1e-4"
        prop_secondary_inter = args.prop_inter2 or "1e-3"
        prop_primary_inter = args.prop_inter or "1e-1"
        default_filler = "1e4"
    else: # mechanics
        prop_A = args.prop_A or "3.0 1.0"
        prop_B = args.prop_B or "3.0 1.0"
        prop_secondary_inter = args.prop_inter2 or "10.0 3.0"
        prop_primary_inter = args.prop_inter or "15.0 5.0"
        default_filler = "100.0 50.0"

    # Count valid recipes
    valid_recipes = [r for r in args.recipe if float(r.split(':')[1]) > 0]
    
    # Fixed ID assignment based on new micro_builder logic
    secondary_inter_id = 2
    primary_inter_id = 3
    filler_start_id = 4
    current_filler_id = filler_start_id
    
    # Initialize property mapping dictionary
    prop_map = {
        0: prop_A, 
        1: prop_B, 
        secondary_inter_id: prop_secondary_inter,
        primary_inter_id: prop_primary_inter
    }
    
    print(f"--- Pipeline Start: {args.basename} ({args.physics_mode} mode, bg: {args.bg_type}, solver: {args.solver}) ---")
    
    build_log = f"{args.basename}_build.log"
    with open(build_log, "w", encoding="utf-8") as f:
        f.write(f"--- Build Log for {args.basename} ---\n")
        f.write(f"Size: {args.size}^3, Voxel: {args.voxel_size}m, BG: {args.bg_type}\n")
        f.write(f"Mode: {args.physics_mode}, Solver: {args.solver}\n")
        f.write(f"Recipe: {' '.join(args.recipe)}\n")

    step_logs = []

    # 1. Branch for background generation
    t0 = time.time()
    if args.bg_type == "single":
        _, tpms_grid, _, actual_phaseA = build_single_phase_grid(args.size)
    elif args.bg_type == "lamellar":
        _, tpms_grid, _, actual_phaseA = build_lamellar_grid(args.size, wavelength=10, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "cylinder":
        _, tpms_grid, _, actual_phaseA = build_cylinder_hex_grid(args.size, wavelength=15, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "bcc":
        _, tpms_grid, _, actual_phaseA = build_bcc_grid(args.size, wavelength=15, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "sea_island":
        _, tpms_grid, _, actual_phaseA = build_sea_island_grid(args.size, island_radius=8, target_phaseA_ratio=args.phaseA_ratio)
    elif args.bg_type == "island_sea":
        _, tpms_grid, _, actual_phaseA = build_island_sea_grid(args.size, island_radius=8, target_phaseA_ratio=args.phaseA_ratio)
    else:
        # Gyroid
        _, tpms_grid, _, actual_phaseA = build_tpms_grid_with_target_ratio(args.size, wavelength=10, target_phaseA_ratio=args.phaseA_ratio)
        
    # Shell counter for electrical/mechanics mode (pass None for thermal)
    comp_grid = np.zeros((args.size, args.size, args.size), dtype=np.uint8)
    shell_count_grid = np.zeros_like(comp_grid) if args.physics_mode in ['electrical', 'mechanics'] else None
    step_logs.append(f"BG({args.bg_type}):{time.time() - t0:.1f}s")

    # Global placement registry for the reference configuration
    placement_registry = []

    # 2. Filler placement based on recipe
    for step in valid_recipes:
        parts = step.split(':')
        f_type = parts[0]
        f_vol = float(parts[1])

        opts = {}
        for p in parts[2:]:
            k, v = p.split('=')
            if k == 'prop': 
                opts[k] = v.replace('_', ' ')
            else: 
                # Parse numeric values robustly, handling scientific notation (e.g., 1e-3)
                try:
                    val = float(v)
                    # Convert to int only if it's a perfect integer without decimal or exponent notation
                    if val.is_integer() and not any(c in v.lower() for c in ['.', 'e']):
                        opts[k] = int(val)
                    else:
                        opts[k] = val
                except ValueError:
                    # Fallback to raw string if it's not a number
                    opts[k] = v

        prop_map[current_filler_id] = opts.pop('prop', default_filler)
        protrusion_coef = opts.pop('protrusion_coef', 0.0025)
        t_step = time.time()
        
        hybrid_args = {
            'comp_grid': comp_grid,
            'tpms_grid': tpms_grid,
            'target_vol_frac': f_vol,
            'protrusion_coef': protrusion_coef,
            'log_file': build_log,
            'physics_mode': args.physics_mode,
            'shell_count_grid': shell_count_grid,
            'filler_id': current_filler_id,
            'inter_id': primary_inter_id, 
            'tunnel_radius': args.tunnel_radius,
            'placement_registry': placement_registry
        }
        
        if f_type == "flake":
            kwargs = {'radius': opts.get('radius', 10), 'thickness': opts.get('thickness', 2)}
            place_fillers_hybrid(filler_func=get_flake_mask, kwargs=kwargs, desc="Flake", **hybrid_args)
                                 
        elif f_type == "sphere":
            kwargs = {'radius': opts.get('radius', 5)}
            place_fillers_hybrid(filler_func=get_sphere_mask, kwargs=kwargs, desc="Sphere", **hybrid_args)
                                 
        elif f_type == "rigidfiber":
            kwargs = {'length': opts.get('length', 60), 'radius': opts.get('radius', 2)}
            place_fillers_hybrid(filler_func=get_rigid_cylinder_mask, kwargs=kwargs, desc="Rigid Fiber", **hybrid_args)
            
        elif f_type == "adaptfiber":
            # NOTE: adaptfiber is explicitly excluded from affine deformation support. 
            # It writes directly to comp_grid and is only evaluated at lambda=1.0.
            place_adaptive_fibers(comp_grid=comp_grid, tpms_grid=tpms_grid, target_vol_frac=f_vol,
                                  length=opts.get('length', 90), radius=opts.get('radius', 2),
                                  max_bend_deg=opts.get('max_bend_deg', 45), max_total_bends=opts.get('max_total_bends', 5),
                                  min_backbone_ratio=opts.get('min_backbone_ratio', 0.9), max_protrusion_ratio=protrusion_coef,
                                  log_file=build_log, physics_mode=args.physics_mode, shell_count_grid=shell_count_grid,
                                  filler_id=current_filler_id, inter_id=primary_inter_id,
                                  tunnel_radius=args.tunnel_radius)
                                  
        elif f_type == "flexfiber":
            kwargs = {
                'length': opts.get('length', 90), 'radius': opts.get('radius', 2),
                'max_bend_deg': opts.get('max_bend_deg', 90), 'max_total_bends': opts.get('max_total_bends', 10)
            }
            place_fillers_hybrid(filler_func=get_flexible_fiber_mask, kwargs=kwargs, desc="Flexible Fibers", **hybrid_args)
                                 
        elif f_type == "agglomerate":
            kwargs = {
                'num_fibers': opts.get('num_fibers', 5), 'length': opts.get('length', 90), 'radius': opts.get('radius', 2),
                'max_bend_deg': opts.get('max_bend_deg', 90), 'max_total_bends': opts.get('max_total_bends', 10)
            }
            desc_str = f"Agglomerate(n={int(opts.get('num_fibers', 5))})"
            place_fillers_hybrid(filler_func=get_agglomerate_mask, kwargs=kwargs, desc=desc_str, **hybrid_args)
                                 
        elif f_type == "staggered":
            kwargs = {
                'radius': opts.get('radius', 15), 'layer_thickness': opts.get('layer_thickness', 2),
                'min_layers': opts.get('min_layers', 1), 'max_layers': opts.get('max_layers', 4),
                'max_offset_pct': opts.get('max_offset_pct', 30)
            }
            place_fillers_hybrid(filler_func=get_staggered_flakes_mask, kwargs=kwargs, desc="Staggered Flakes", **hybrid_args)

        step_logs.append(f"{f_type}(ID:{current_filler_id}):{time.time() - t_step:.1f}s")
        current_filler_id += 1

    # =========================================================================
    # 3. One-Shot Stretching & Evaluation Loop
    # =========================================================================
    
    pvd_records = []
    pvd_filename = f"{args.basename}.pvd"
    
    # Ensure apply_background_deformation and render_deformed_fillers are imported at the top of run_pipeline.py!
    
    for stretch in args.stretch_ratios:
        print(f"\n{'='*50}")
        print(f"--- Processing Stretch Ratio: {stretch} ---")
        print(f"{'='*50}")
        
        current_basename = f"{args.basename}_L{stretch:.2f}"
        current_step_logs = step_logs.copy()
        t_def = time.time()
        
        # Deform ONLY the continuous polymer matrix
        current_tpms = apply_background_deformation(tpms_grid, stretch, args.poisson_ratio)
        
        # Initialize empty spaces for rigid fillers
        current_comp = np.zeros_like(current_tpms)
        current_shell = np.zeros_like(current_tpms) if args.physics_mode in ['electrical', 'mechanics'] else None
        
        # Redraw all rigid fillers using kinematic transformations
        render_deformed_fillers(
            placement_registry, 
            base_shape=(args.size, args.size, args.size), 
            stretch_ratio=stretch, 
            poisson_ratio=args.poisson_ratio, 
            is_thermal=(args.physics_mode == 'thermal'), 
            comp_grid=current_comp, 
            shell_count_grid=current_shell, 
            tunnel_radius=args.tunnel_radius
        )
        
        current_step_logs.append(f"Deformation({stretch}):{time.time() - t_def:.1f}s")

        # 4. Integration of final structure and creation of metadata
        final_grid = finalize_microstructure(
            current_comp, current_tpms, current_shell, args.physics_mode, 
            secondary_inter_id=secondary_inter_id, 
            primary_inter_id=primary_inter_id, 
            filler_start_id=filler_start_id,
            contact_radius=args.contact_radius
        )
        
        phase_stats = summarize_phase_fractions(
            final_grid, secondary_inter_id=secondary_inter_id, primary_inter_id=primary_inter_id, filler_start_id=filler_start_id
        )
        if args.skip_structure_metrics:
            structure_metrics = {
                'contact_ratio': 0.0, 'tunneling_ratio': 0.0, 'connectivity_ratio': 0.0,
                'n_contact_voxels': 0, 'n_tunnel_voxels': 0, 'n_filler_voxels': 0,
                'n_conductive_candidate_voxels': 0, 'n_largest_cluster_voxels': 0, 'n_conductive_clusters': 0,
            }
        else:
            structure_metrics = compute_structure_metrics(
                final_grid,
                primary_inter_id=primary_inter_id,
                secondary_inter_id=secondary_inter_id,
                filler_start_id=filler_start_id
            )
        metric_fields = structure_metrics_to_csv_fields(structure_metrics)

        metadata = {
            "Basename": current_basename,
            "Grid_Size": f"{final_grid.shape[2]}x{final_grid.shape[1]}x{final_grid.shape[0]}",
            "Voxel_Size_m": str(args.voxel_size),
            "BG_Type": args.bg_type,
            "Physics_Mode": args.physics_mode,
            "Solver": args.solver,
            "Recipe": " ".join(args.recipe),
            "PolymerA_Frac": f"{phase_stats['polymer_a_fraction']:.4f}",
            "PolymerB_Frac": f"{phase_stats['polymer_b_fraction']:.4f}",
            "Secondary_Inter_Frac": f"{phase_stats['secondary_interface_fraction']:.4f}",
            "Primary_Inter_Frac": f"{phase_stats['primary_interface_fraction']:.4f}",
            "Filler_Frac": f"{phase_stats['filler_total_fraction']:.4f}",
            "Contact_Ratio": metric_fields["Contact_Ratio"],
            "Tunneling_Ratio": metric_fields["Tunneling_Ratio"],
            "Connectivity_Ratio": metric_fields["Connectivity_Ratio"],
            "Stretch_Ratio": str(stretch),
            "Poisson_Ratio": str(args.poisson_ratio) if stretch != 1.0 else "N/A",
        }
        
        # Export VTI for visualization (in the specified directory)
        vti_filename = f"{current_basename}.vti"
        export_visualization_vti(final_grid, vti_filename, voxel_size=args.voxel_size, metadata=metadata)
        
        # --- Create VTM wrapper and update PVD in the 'vtm/' subdirectory ---
        # Extract the directory path from current_basename
        base_dir = os.path.dirname(current_basename)
        file_stem = os.path.basename(current_basename)
        base_stem = os.path.basename(args.basename)
        
        # Determine the target vtm directory (one level below the VTI file)
        wrapper_dir = os.path.join(base_dir, "vtm") if base_dir else "vtm"
        vtm_filename = os.path.join(wrapper_dir, f"{file_stem}.vtm")
        export_vtm_wrapper(vti_filename, vtm_filename)
        
        # Use the stretch ratio as the timestep value
        pvd_filename = os.path.join(wrapper_dir, f"{base_stem}.pvd")
        pvd_records.append((stretch, vtm_filename))
        update_pvd_file(pvd_filename, pvd_records)
        print(f"Generated VTM wrapper and updated PVD collection in: {pvd_filename}")

        slice_filename = f"{current_basename}_slice.png"
        save_thumbnail_png(final_grid, slice_filename)

        # 5. Execute computational solvers and aggregate results
        chfem_time = ""
        chfem_results = [""] * 6
        puma_time = ""
        puma_results = [""] * 6

        export_chfem_inputs(final_grid, current_basename, voxel_size=args.voxel_size, physics_mode=args.physics_mode, prop_map=prop_map)
        
        if args.solver in ["chfem", "both"]:
            log_file = f"{current_basename}_metrics.txt"
            subprocess.run(["chfem_exec", f"{current_basename}.nf", f"{current_basename}.raw", "-m", log_file])
            res_diag, ctime = parse_chfem_log(log_file)
            if res_diag[0] != "":
                chfem_time, chfem_results = f"{ctime:.2f}", res_diag

        if args.solver in ["puma", "both"]:
            cond_map = {k: float(v) for k, v in prop_map.items()}
            pkx, pky, pkz, ptime = run_puma_laplace(final_grid, args.voxel_size, args.physics_mode, cond_map)
            if pkx is not None:
                puma_time = f"{ptime:.2f}"
                # PuMA (Laplace) only provides 3 diagonal components
                puma_results = [pkx, pky, pkz, "", "", ""]

        file_exists = os.path.isfile(args.csv_log)
        with open(args.csv_log, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Header with universal T-notation
                writer.writerow([
                    "Basename", "Grid_Size", "Voxel_Size_m", "BG_Type", "Mode", "Solver", "Recipe", 
                    "Stretch_Ratio", "Poisson_Ratio",
                    "PolymerA_Frac", "PolymerB_Frac", "Secondary_Inter_Frac", "Primary_Inter_Frac", "Filler_Frac",
                    "Contact_Ratio", "Tunneling_Ratio", "Connectivity_Ratio",
                    "N_Contact_Voxels", "N_Tunnel_Voxels", "N_Filler_Voxels", "N_Conductive_Candidate_Voxels",
                    "N_Largest_Cluster_Voxels", "N_Conductive_Clusters",
                    "Placement_Logs", "Slice_Image",
                    "chfem_Time_s", "chfem_Txx", "chfem_Tyy", "chfem_Tzz", "chfem_Tyz", "chfem_Tzx", "chfem_Txy",
                    "puma_Time_s", "puma_Txx", "puma_Tyy", "puma_Tzz", "puma_Tyz", "puma_Tzx", "puma_Txy"
                ])
                
            writer.writerow([
                current_basename, f"{final_grid.shape[2]}x{final_grid.shape[1]}x{final_grid.shape[0]}", args.voxel_size, args.bg_type, args.physics_mode, args.solver, " ".join(args.recipe),
                stretch, args.poisson_ratio if stretch != 1.0 else "N/A",
                f"{phase_stats['polymer_a_fraction']:.4f}",
                f"{phase_stats['polymer_b_fraction']:.4f}",
                f"{phase_stats['secondary_interface_fraction']:.4f}", 
                f"{phase_stats['primary_interface_fraction']:.4f}", 
                f"{phase_stats['filler_total_fraction']:.4f}",
                metric_fields["Contact_Ratio"], metric_fields["Tunneling_Ratio"], metric_fields["Connectivity_Ratio"],
                metric_fields["N_Contact_Voxels"], metric_fields["N_Tunnel_Voxels"], metric_fields["N_Filler_Voxels"],
                metric_fields["N_Conductive_Candidate_Voxels"], metric_fields["N_Largest_Cluster_Voxels"], metric_fields["N_Conductive_Clusters"],
                " | ".join(current_step_logs),
                slice_filename,
                chfem_time, *chfem_results,
                puma_time, *puma_results
            ])
                

if __name__ == "__main__":
    main()
