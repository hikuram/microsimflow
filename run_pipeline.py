import argparse
import csv
import glob
import os
import subprocess
import time

import numpy as np

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
    get_flake_mask,
    get_sphere_mask,
    get_rigid_cylinder_mask,
    get_irregular_fiber_mask,
    get_flexible_fiber_mask,
    get_agglomerate_mask,
    get_staggered_flakes_mask,
    finalize_microstructure,
    export_chfem_inputs,
    export_visualization_vti,
    summarize_phase_fractions,
    compute_structure_metrics,
    compute_advanced_metrics,
    apply_background_deformation,
    render_deformed_fillers
)

from pipeline.io_csv import (
    ensure_structure_metric_columns,
    parse_chfem_log,
    structure_metrics_to_csv_fields,
    upgrade_existing_csv_log,
)
from pipeline.solver_puma import run_puma_elasticity, run_puma_laplace, run_puma_permeability
from pipeline.viz_export import (
    export_common_legend,
    export_vtm_wrapper,
    save_thumbnail_png,
    update_pvd_file,
)

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
    parser.add_argument("--base_rotation", type=int, default=0, choices=[0, 1, 2],
                        help="Background rotation mode before filler placement (0: None, 1: YZX, 2: ZXY). Default: 0.")
    parser.add_argument("--phaseA_ratio", type=float, default=0.50, help="Target volume fraction for Phase A in the background (default: 0.50).")
    parser.add_argument("--feature_size", type=float, default=10.0, help="Characteristic length (wavelength for TPMS, radius for Islands). Default: 10.0")
    parser.add_argument("--diffusion_factor", type=float, default=0.0, help="Mutual diffusion factor for background phases (0.0 = sharp).")

    group = parser.add_mutually_exclusive_group(required=True)
    
    recipe_help = """Recipe for filler placement (required for new build).
Format: type:volume_fraction:param1=val1:param2=val2...
Available types and common parameters:
  - flake        : radius, thickness
  - sphere       : radius
  - rigidfiber   : length, radius
  - flexfiber    : length, radius, max_bend_deg, max_total_bends
  - agglomerate  : num_fibers, length, radius, max_bend_deg, max_total_bends
  - staggered    : radius, layer_thickness, min_layers, max_layers, max_offset_pct
Optional param: 'prop=X' to override the physical property for this specific filler.
Example: --recipe "rigidfiber:0.05:length=60:radius=2:prop=500.0" "flake:0.02:radius=15:thickness=2"
"""
    group.add_argument("--recipe", nargs='+', help=recipe_help)
    parser.add_argument("--basename", type=str, default="model", help="Base filename for generated files (default: 'model').")
    parser.add_argument("--csv_log", type=str, default="comparison_results.csv", help="CSV file to append/update results (default: 'comparison_results.csv').")
    parser.add_argument("--writer", type=str, default="vti", choices=["vti", "zstd", "arrow"],
                        help="Export format for the final microstructure (vti, zstd, arrow). Default: vti.")
    parser.add_argument("--vti_fields", type=str, default="off", choices=["on", "off"],
                        help="Embed additional physical fields (pressure, velocity, etc.) into VTI (default: off).")
    parser.add_argument("--physics_mode", type=str, default="thermal",
                        choices=["thermal", "electrical", "mechanics", "permeability"],
                        help="Physics mode which defines interface handling and default properties (default: thermal).")
    parser.add_argument("--void_phases", type=int, nargs='+', default=[0], 
                        help="Phase IDs to be treated as void (fluid) in permeability mode. Default is Phase 0 (Polymer A).")
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
                        help="List of stretch ratios (lambda) along the specified axis to evaluate deformation (default: 1.0).")
    parser.add_argument("--stretch_axis", type=str, default="X", choices=["X", "Y", "Z"],
                        help="Direction of uniaxial stretch (X, Y, or Z). Default: X.")
    parser.add_argument("--poisson_ratio", type=float, default=0.4, 
                        help="Poisson's ratio (nu) for transverse compression during deformation (default: 0.4).")
    parser.add_argument("--deformation_mode", type=str, default="fine", choices=["coarse", "fine"],
                        help="Filler rendering mode for stretch ratios other than 1.0 (default: fine).")
    parser.add_argument("--fine_volume_tol", type=float, default=0.01,
                        help="Per-placement added-volume tolerance for fine deformation mode (default: 0.01).")
    parser.add_argument("--fine_max_tilt_deg", type=float, default=0.10,
                        help="Maximum two-axis tilt correction in degrees for fine deformation mode (default: 0.10).")
    parser.add_argument("--fine_ledger_cap", type=float, default=0.01,
                        help="Maximum per-placement ledger correction ratio in fine deformation mode (default: 0.01).")
    parser.add_argument("--skip_structure_metrics", action="store_true", help="Skip lightweight post-process structure metrics (enabled by default).")
    parser.add_argument("--advanced_metrics", action="store_true", help="Calculate advanced PoreSpy metrics (SSA, Local Thickness, Autocorrelation).")
    return parser.parse_args()

def main():
    args = parse_args()
    
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
        prop_secondary_inter = args.prop_inter2 or "3.0"
        prop_primary_inter = args.prop_inter or "30.0"
        default_filler = "300.0"
    elif args.physics_mode == 'electrical':
        prop_A = args.prop_A or "1e-4"
        prop_B = args.prop_B or "1e-4"
        prop_secondary_inter = args.prop_inter2 or "1e-3"
        prop_primary_inter = args.prop_inter or "1e-1"
        default_filler = "1e4"
    else: # mechanics, permeability
        prop_A = args.prop_A or "1.0 0.35"
        prop_B = args.prop_B or "1.0 0.35"
        prop_secondary_inter = args.prop_inter2 or "10.0 0.30"
        prop_primary_inter = args.prop_inter or "100.0 0.25"
        default_filler = "1000.0 0.20"

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
        _, tpms_grid, _, actual_phaseA = build_single_phase_grid(args.size, feature_size=args.feature_size, diffusion_factor=args.diffusion_factor)
    elif args.bg_type == "lamellar":
        _, tpms_grid, _, actual_phaseA = build_lamellar_grid(args.size, feature_size=args.feature_size, target_phaseA_ratio=args.phaseA_ratio, diffusion_factor=args.diffusion_factor)
    elif args.bg_type == "cylinder":
        _, tpms_grid, _, actual_phaseA = build_cylinder_hex_grid(args.size, feature_size=args.feature_size, target_phaseA_ratio=args.phaseA_ratio, diffusion_factor=args.diffusion_factor)
    elif args.bg_type == "bcc":
        _, tpms_grid, _, actual_phaseA = build_bcc_grid(args.size, feature_size=args.feature_size, target_phaseA_ratio=args.phaseA_ratio, diffusion_factor=args.diffusion_factor)
    elif args.bg_type == "sea_island":
        _, tpms_grid, _, actual_phaseA = build_sea_island_grid(args.size, feature_size=args.feature_size, target_phaseA_ratio=args.phaseA_ratio, diffusion_factor=args.diffusion_factor)
    elif args.bg_type == "island_sea":
        _, tpms_grid, _, actual_phaseA = build_island_sea_grid(args.size, feature_size=args.feature_size, target_phaseA_ratio=args.phaseA_ratio, diffusion_factor=args.diffusion_factor)
    else:
        # Gyroid
        _, tpms_grid, _, actual_phaseA = build_tpms_grid_with_target_ratio(args.size, feature_size=args.feature_size, target_phaseA_ratio=args.phaseA_ratio, diffusion_factor=args.diffusion_factor)
        
    # Initialize comp_grid with unsigned 8-bit integer for memory efficiency
    comp_grid = np.zeros((args.size, args.size, args.size), dtype=np.uint8)
    
    # Initialize shell_count_grid unconditionally for ALL modes.
    # In Thermal mode, we will use bitwise operations on this uint8 array to track both overlaps and shells.
    shell_count_grid = np.zeros_like(comp_grid)
    
    # -------------------------------------------------------------------------
    # Background Base Rotation
    # Apply cyclic permutation to the discrete grid before any fillers are placed.
    # This maintains the right-handed coordinate system and prevents any
    # interpolation or shape distortion (jagged edges, volume fraction loss)
    # that would occur with arbitrary angle rotations.
    # -------------------------------------------------------------------------
    if args.base_rotation == 1:
        # Mode 1: YZX cyclic permutation
        tpms_grid = np.ascontiguousarray(np.transpose(tpms_grid, axes=(1, 2, 0)))
        comp_grid = np.ascontiguousarray(np.transpose(comp_grid, axes=(1, 2, 0)))
        shell_count_grid = np.ascontiguousarray(np.transpose(shell_count_grid, axes=(1, 2, 0)))
    elif args.base_rotation == 2:
        # Mode 2: ZXY cyclic permutation
        tpms_grid = np.ascontiguousarray(np.transpose(tpms_grid, axes=(2, 0, 1)))
        comp_grid = np.ascontiguousarray(np.transpose(comp_grid, axes=(2, 0, 1)))
        shell_count_grid = np.ascontiguousarray(np.transpose(shell_count_grid, axes=(2, 0, 1)))
    
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
            # --- Vector parsing for orientation control ---
            elif k == 'mean_dir':
                opts[k] = [float(x) for x in v.split(',')]
            # ----------------------------------------------
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
            'tunnel_radius': args.tunnel_radius,
            'placement_registry': placement_registry
        }
        
        if f_type == "flake":
            kwargs = {
                'radius': opts.get('radius', 10), 'thickness': opts.get('thickness', 2),
                'mean_dir': opts.get('mean_dir', [0.0, 0.0, 1.0]), 'kappa': opts.get('kappa', 0.0)
            }
            place_fillers_hybrid(filler_func=get_flake_mask, kwargs=kwargs, desc="Flake", **hybrid_args)
                                 
        elif f_type == "sphere":
            kwargs = {'radius': opts.get('radius', 5)}
            place_fillers_hybrid(filler_func=get_sphere_mask, kwargs=kwargs, desc="Sphere", **hybrid_args)
                                 
        elif f_type == "rigidfiber":
            kwargs = {
                'length': opts.get('length', 60), 'radius': opts.get('radius', 2),
                'mean_dir': opts.get('mean_dir', [0.0, 0.0, 1.0]), 'kappa': opts.get('kappa', 0.0)
            }
            place_fillers_hybrid(filler_func=get_rigid_cylinder_mask, kwargs=kwargs, desc="Rigid Fiber", **hybrid_args)
            
        elif f_type == "irregfiber":
            kwargs = {
                'length': opts.get('length', 60),
                'shape_type': opts.get('shape', 'bean'), 'radius_max': opts.get('radius', 5), 'ratio': opts.get('ratio', 0.5),
                'mean_dir': opts.get('mean_dir', [0.0, 0.0, 1.0]), 'kappa': opts.get('kappa', 0.0)
            }
            place_fillers_hybrid(filler_func=get_irregular_fiber_mask, kwargs=kwargs, desc="Irregular Fiber", **hybrid_args)
        
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
                'max_offset_pct': opts.get('max_offset_pct', 30),
                'mean_dir': opts.get('mean_dir', [0.0, 0.0, 1.0]), 'kappa': opts.get('kappa', 0.0)
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
        
        if np.isclose(stretch, 1.0):
            # Bypass all deformation and re-rendering at lambda = 1.0.
            current_tpms = tpms_grid.copy()
            current_comp = comp_grid.copy()
            current_shell = shell_count_grid.copy()
        else:
            # Deform ONLY the continuous polymer matrix
            current_tpms = apply_background_deformation(tpms_grid, stretch, args.poisson_ratio, args.stretch_axis)
            
            # Initialize empty spaces for rigid fillers
            current_comp = np.zeros_like(current_tpms)
            
            # Initialize shell grid unconditionally to re-render shells and overlaps during stretching
            current_shell = np.zeros_like(current_tpms)
            
            # Redraw all rigid fillers using kinematic transformations
            render_deformed_fillers(
                placement_registry,
                base_shape=(args.size, args.size, args.size),
                stretch_ratio=stretch,
                poisson_ratio=args.poisson_ratio,
                comp_grid=current_comp, 
                shell_count_grid=current_shell,
                stretch_axis=args.stretch_axis,
                tunnel_radius=args.tunnel_radius,
                deformation_mode=args.deformation_mode,
                fine_volume_tol=args.fine_volume_tol,
                fine_max_tilt_deg=args.fine_max_tilt_deg,
                fine_ledger_cap=args.fine_ledger_cap
            )
        
        current_step_logs.append(f"Deformation({stretch},{args.deformation_mode}):{time.time() - t_def:.1f}s")

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
            
            # --- Evaluate advanced PoreSpy metrics if requested ---
            if args.advanced_metrics:
                adv_metrics = compute_advanced_metrics(final_grid, args.voxel_size, filler_start_id=filler_start_id)
                structure_metrics.update(adv_metrics)
                
        metric_fields = structure_metrics_to_csv_fields(structure_metrics)

        metadata = {
            "Basename": current_basename,
            "Grid_Size": f"{final_grid.shape[2]}x{final_grid.shape[1]}x{final_grid.shape[0]}",
            "Voxel_Size_m": str(args.voxel_size),
            "BG_Type": args.bg_type,
            "Base_Rotation": str(args.base_rotation),
            "Physics_Mode": args.physics_mode,
            "Solver": args.solver,
            "Recipe": " ".join(args.recipe),
            "Stretch_Ratio": str(stretch),
            "Stretch_Axis": args.stretch_axis,
            "PolymerA_Frac": f"{phase_stats['polymer_a_fraction']:.4f}",
            "PolymerB_Frac": f"{phase_stats['polymer_b_fraction']:.4f}",
            "Secondary_Inter_Frac": f"{phase_stats['secondary_interface_fraction']:.4f}",
            "Primary_Inter_Frac": f"{phase_stats['primary_interface_fraction']:.4f}",
            "Filler_Frac": f"{phase_stats['filler_total_fraction']:.4f}",
            "Contact_Ratio": metric_fields["Contact_Ratio"],
            "Tunneling_Ratio": metric_fields["Tunneling_Ratio"],
            "Connectivity_Ratio": metric_fields["Connectivity_Ratio"],
            "Poisson_Ratio": str(args.poisson_ratio) if stretch != 1.0 else "N/A",
        }
        
        # Export VTI for visualization (in the specified directory)
        # Determine output extension based on selected writer
        ext_map = {'vti': 'vti', 'zstd': 'pv', 'arrow': 'arrow'}
        out_ext = ext_map.get(args.writer, 'vti')
        out_filename = f"{current_basename}.{out_ext}"
        
        # Export voxel grid for visualization / storage
        export_visualization_vti(final_grid, out_filename, voxel_size=args.voxel_size, metadata=metadata, writer=args.writer)
        
        # --- Create VTM wrapper and update PVD in the 'vtm/' subdirectory ---
        # Note: ParaView's PVD/VTM collections natively support VTK XML formats (.vti).
        # We skip VTM/PVD generation for non-VTI formats to prevent ParaView loader errors.
        if args.writer == "vti":
            base_dir = os.path.dirname(current_basename)
            file_stem = os.path.basename(current_basename)
            base_stem = os.path.basename(args.basename)
            
            wrapper_dir = os.path.join(base_dir, "vtm") if base_dir else "vtm"
            vtm_filename = os.path.join(wrapper_dir, f"{file_stem}.vtm")
            export_vtm_wrapper(out_filename, vtm_filename)
            
            pvd_filename = os.path.join(wrapper_dir, f"{base_stem}.pvd")
            pvd_records.append((stretch, vtm_filename))
            update_pvd_file(pvd_filename, pvd_records)
            print(f"Generated VTM wrapper and updated PVD collection in: {pvd_filename}")
        else:
            print(f"Skipped PVD collection update (Format '{args.writer}' is optimized for Python/ML pipelines).")

        slice_filename = f"{current_basename}_slice.png"
        save_thumbnail_png(final_grid, slice_filename)

        # 5. Execute computational solvers and aggregate results
        chfem_time = ""
        chfem_results = [""] * 6
        puma_time = ""
        puma_results = [""] * 6
        
        # --- Binarization for permeability mode ---
        if args.physics_mode == 'permeability':
            print(f"  -> Binarizing grid for permeability mode (Void phases: {args.void_phases})")
            export_grid = np.zeros_like(final_grid)
            for vp in args.void_phases:
                export_grid[final_grid == vp] = 1
            export_prop_map = {0: "0.0", 1: "1.0"} 
        else:
            export_grid = final_grid
            export_prop_map = prop_map
            
        export_chfem_inputs(export_grid, current_basename, voxel_size=args.voxel_size, physics_mode=args.physics_mode, prop_map=export_prop_map)
        
        if args.solver in ["chfem", "both"]:
            log_file = f"{current_basename}_metrics.txt"
            
            # Dynamically build the command. Add '-e' (export) only if VTI fields are requested.
            chfem_cmd = ["chfem_exec", f"{current_basename}.nf", f"{current_basename}.raw", "-m", log_file]
            if args.vti_fields == "on":
                chfem_cmd.insert(3, "-e")
                
            subprocess.run(chfem_cmd)
            
            res_diag, ctime = parse_chfem_log(log_file)
            if res_diag[0] != "":
                chfem_time, chfem_results = f"{ctime:.2f}", res_diag

        if args.solver in ["puma", "both"]:
            if args.physics_mode == "mechanics":
                puma_results, ptime = run_puma_elasticity(final_grid, args.voxel_size, prop_map)
                if puma_results[0] is not None:
                    puma_time = f"{ptime:.2f}"
        
            elif args.physics_mode == "permeability":
                puma_results, ptime = run_puma_permeability(export_grid, args.voxel_size, solid_cutoff=(0, 0))
                if puma_results[0] is not None:
                    puma_time = f"{ptime:.2f}"
                    
            else:
                cond_map = {k: float(v.split()[0]) for k, v in prop_map.items()}
                pkx, pky, pkz, ptime = run_puma_laplace(final_grid, args.voxel_size, args.physics_mode, cond_map)
                if pkx is not None:
                    puma_time = f"{ptime:.2f}"
                    # PuMA (Laplace) only provides 3 diagonal components
                    puma_results = [pkx, pky, pkz, "", "", ""]
        
        # =====================================================================
        # --- Extract and embed physical fields into VTI ---
        # =====================================================================
        extra_viz_fields = {}
        if args.vti_fields == "on" and args.solver in ["chfem", "both"]:
            print(f"  -> Scanning for physical field outputs to embed in VTI...")

            # Define base filenames and the corresponding chfem binary suffixes and components.
            # Note: chfem outputs "temperature" and "flux" binaries for both thermal and electrical analyses.
            field_mappings = {
                'permeability': {'Pressure': ("pressure", 1), 'Velocity': ("velocity", 3)},
                'thermal': {'Temperature': ("temperature", 1), 'Heat_Flux': ("flux", 3)},
                'electrical': {'Potential': ("temperature", 1), 'Electric_Current': ("flux", 3)},
                'mechanics': {'Displacement': ("displacement", 3), 'Stress': ("stress", 6)}
            }

            if args.physics_mode in field_mappings:
                found_any = False
                nz, ny, nx = final_grid.shape
                
                dir_labels = ['X', 'Y', 'Z', 'YZ', 'ZX', 'XY']
                
                for base_field_name, (file_suffix, comps) in field_mappings[args.physics_mode].items():
                    for dir_idx in range(6):
                        file_path = f"{current_basename}_{file_suffix}_{dir_idx}.bin"
                        if os.path.exists(file_path):
                            found_any = True
                            dir_label = dir_labels[dir_idx]
                            
                            # Append macro-direction to the field name (e.g., Potential_X, Electric_Current_Y)
                            field_name = f"{base_field_name}_{dir_label}"
                            
                            try:
                                raw_data = np.fromfile(file_path, dtype=np.float64)
                                
                                # As per chfem/io.py, data is exported in (Z, X, Y, comps) order.
                                # Restore spatial axes to NumPy's (Z, Y, X) without altering component order.
                                if comps == 1:
                                    field_data = raw_data.reshape((nz, nx, ny)).transpose(0, 2, 1)
                                    extra_viz_fields[field_name] = field_data
                                else:
                                    field_data = raw_data.reshape((nz, nx, ny, comps)).transpose(0, 2, 1, 3)
                                    
                                    # === Forcefully split all vectors and tensors into independent scalars ===
                                    if comps == 3:
                                        # Workaround: ParaView's Volume Rendering often ignores Y/Z vector channel selections.
                                        vec_labels = ['x', 'y', 'z']
                                        for i, vlabel in enumerate(vec_labels):
                                            extra_viz_fields[f"{field_name}_{vlabel}"] = field_data[..., i]
                                            
                                    elif comps == 6:
                                        # Workaround: ParaView's Volume Mapper crashes on arrays with >4 components.
                                        stress_labels = ['xx', 'yy', 'zz', 'yz', 'zx', 'xy']
                                        for i, slabel in enumerate(stress_labels):
                                            extra_viz_fields[f"{field_name}_{slabel}"] = field_data[..., i]
                                            
                                print(f"    - Imported field: {field_name} from {file_path}")
                            except Exception as e:
                                print(f"    - Failed to import field {field_name}: {e}")
                
                if not found_any:
                    print("    ! Warning: No binary output files were found.")
                    print("    ! Make sure chfem_exec is executed with the export flag (e.g., '-e').")

            # Re-export voxel grid with the extra fields included
            if extra_viz_fields:
                export_visualization_vti(
                    final_grid, out_filename, 
                    voxel_size=args.voxel_size, 
                    metadata=metadata, 
                    extra_fields=extra_viz_fields,
                    writer=args.writer
                )
                print(f"  -> Updated {args.writer.upper()} output with physical fields: {out_filename}")
                
            # --- CLEANUP: Remove raw binary files generated by chfem ---
            bin_files = glob.glob(f"{current_basename}_*.bin")
            for b_file in bin_files:
                try:
                    os.remove(b_file)
                    print(f"    - Cleaned up temp binary file: {b_file}")
                except OSError as e:
                    print(f"    - Failed to delete {b_file}: {e}")

        file_exists = os.path.isfile(args.csv_log)
        with open(args.csv_log, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Header with universal T-notation and advanced metrics
                writer.writerow([
                    "Basename", "Grid_Size", "Voxel_Size_m", "BG_Type", "Base_Rotation", "Mode", "Solver", "Recipe", 
                    "Stretch_Ratio", "Stretch_Axis", "Poisson_Ratio",
                    "PolymerA_Frac", "PolymerB_Frac", "Secondary_Inter_Frac", "Primary_Inter_Frac", "Filler_Frac",
                    "Contact_Ratio", "Tunneling_Ratio", "Connectivity_Ratio",
                    "N_Contact_Voxels", "N_Tunnel_Voxels", "N_Filler_Voxels", "N_Conductive_Candidate_Voxels",
                    "N_Largest_Cluster_Voxels", "N_Conductive_Clusters",
                    "Specific_Surface_Area", "Mean_Sphericity", "Local_Thickness_Mean", "Autocorrelation_Length",
                    "Placement_Logs", "Slice_Image",
                    "chfem_Time_s", "chfem_Txx", "chfem_Tyy", "chfem_Tzz", "chfem_Tyz", "chfem_Tzx", "chfem_Txy",
                    "puma_Time_s", "puma_Txx", "puma_Tyy", "puma_Tzz", "puma_Tyz", "puma_Tzx", "puma_Txy"
                ])
                
            writer.writerow([
                current_basename, f"{final_grid.shape[2]}x{final_grid.shape[1]}x{final_grid.shape[0]}", args.voxel_size, args.bg_type, args.base_rotation, args.physics_mode, args.solver, " ".join(args.recipe),
                stretch, args.stretch_axis, args.poisson_ratio if stretch != 1.0 else "N/A",
                f"{phase_stats['polymer_a_fraction']:.4f}",
                f"{phase_stats['polymer_b_fraction']:.4f}",
                f"{phase_stats['secondary_interface_fraction']:.4f}", 
                f"{phase_stats['primary_interface_fraction']:.4f}", 
                f"{phase_stats['filler_total_fraction']:.4f}",
                metric_fields["Contact_Ratio"], metric_fields["Tunneling_Ratio"], metric_fields["Connectivity_Ratio"],
                metric_fields["N_Contact_Voxels"], metric_fields["N_Tunnel_Voxels"], metric_fields["N_Filler_Voxels"],
                metric_fields["N_Conductive_Candidate_Voxels"], metric_fields["N_Largest_Cluster_Voxels"], metric_fields["N_Conductive_Clusters"],
                metric_fields.get("Specific_Surface_Area", ""), metric_fields.get("Mean_Sphericity", ""), metric_fields.get("Local_Thickness_Mean", ""), metric_fields.get("Autocorrelation_Length", ""),
                " | ".join(current_step_logs),
                slice_filename,
                chfem_time, *chfem_results,
                puma_time, *puma_results
            ])

if __name__ == "__main__":
    main()
